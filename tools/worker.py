"""RallyVision 分析 worker:从 TennisMatch backend 轮询领任务,本机跑 pipeline,回传结果。

用法 / Usage:
    RV_WORKER_TOKEN=xxx uv run tools/worker.py --once            # 领一单跑完退出(内测手动模式)
    RV_WORKER_TOKEN=xxx uv run tools/worker.py --loop --interval 300

视频与输出留在本地 work_dir 不删,内测期就是调试素材。
"""
import argparse
import json
import os
import subprocess
import sys
import time

import httpx

# 按路径调用(README 文档的 `uv run tools/worker.py`)时 Python 只把本文件所在的
# tools/ 目录塞进 sys.path[0],仓库根目录不在里面,下面的绝对导入会
# ModuleNotFoundError。显式把仓库根目录加进 sys.path,兼容脚本路径调用和
# `python -m tools.worker` 两种起法。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.report_builder import ReportBuildError, build_report, has_highlights

BACKEND_BASE = os.environ.get("RV_BACKEND_BASE", "https://api.letstennis.app")
WORK_ROOT = os.environ.get("RV_WORK_DIR", os.path.expanduser("~/rallyvision-jobs"))

# main.py 白名单参数:job.params 里只有这些键会被映射成 CLI flag
_PARAM_FLAGS = {
    "first_server": "--first-server",
    "best_of": "--best-of",
    "no_ad": "--no-ad",
    "upper_hand": "--upper-hand",
    "lower_hand": "--lower-hand",
}
_DEFAULTS = {"first_server": "lower"}


def build_cli_args(video_path, output_dir, params):
    merged = dict(_DEFAULTS)
    for k, v in (params or {}).items():
        if k in _PARAM_FLAGS:
            merged[k] = v
    args = [
        "uv", "run", "main.py",
        "--video-path", str(video_path),
        "--output-dir", str(output_dir),
        "--ball-detector", "tracknet",
        "--court-calibration", "keypoints",
        "--shot-metrics", "true",
        "--line-call", "singles",       # 阶段 4 第一版只支持单打
        "--match-scoring", "true",
        "--highlights", "true",
        "--display", "false",
        "--visualize-positions", "false",
    ]
    for k, v in merged.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        args.extend([_PARAM_FLAGS[k], str(v)])
    return args


def pick_error_code(exc):
    if isinstance(exc, ReportBuildError):
        return exc.code
    return "pipeline_error"


# pipeline 跑完后三次上报(result POST / highlights PUT / complete POST)的重试策略:
# 只重试瞬时网络故障(TransportError)和 5xx,4xx 是业务错误(如 claim token 失效)不重试。
# retry policy for the three post-pipeline HTTP calls — only transient network errors / 5xx
# are retried; 4xx is a business error (e.g. stale claim token) and raises immediately.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SEC = (5, 15)


def with_retry(fn, *, attempts=_RETRY_ATTEMPTS, backoff=_RETRY_BACKOFF_SEC, sleep_fn=time.sleep):
    """执行 fn(),仅对 httpx.TransportError 与 5xx httpx.HTTPStatusError 重试(共 attempts 次,
    退避 backoff[i] 秒);4xx 或其他异常直接抛出不重试。sleep_fn 可注入以便单测不真等待。

    Run fn(); retry only on httpx.TransportError and 5xx HTTPStatusError (up to `attempts`
    times, sleeping backoff[i] seconds between). 4xx or any other exception raises immediately.
    sleep_fn is injectable so tests don't actually sleep.
    """
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is None or status < 500:
                raise
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc
        if i < attempts - 1:
            sleep_fn(backoff[i])
    raise last_exc


def pipeline_timeout(duration_sec):
    """subprocess 跑 pipeline 的超时秒数:duration_sec*40+1800,封顶 72000(20h),可用
    RV_PIPELINE_TIMEOUT 环境变量整体覆盖(防第三个未知阻塞点把无人值守 --loop worker 挂死)。

    Timeout (seconds) for the pipeline subprocess: duration_sec*40+1800, capped at 72000
    (20h); overridable via RV_PIPELINE_TIMEOUT env var (guards against an unattended --loop
    worker wedging forever on some unknown blocking call).
    """
    override = os.environ.get("RV_PIPELINE_TIMEOUT")
    if override:
        return int(override)
    return min(duration_sec * 40 + 1800, 72000)


def _client():
    token = os.environ.get("RV_WORKER_TOKEN", "")
    if not token:
        print("缺少 RV_WORKER_TOKEN / missing RV_WORKER_TOKEN", file=sys.stderr)
        sys.exit(2)
    return httpx.Client(base_url=BACKEND_BASE, timeout=60.0,
                        headers={"X-Worker-Token": token})


def _download(client, url, dest):
    with client.stream("GET", url) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1024 * 1024):
                f.write(chunk)


def process_one(client):
    """领一单并处理。返回 True=处理了一单,False=队列为空。"""
    job = client.post("/v1/rallyvision/worker/claim").raise_for_status().json()["job"]
    if job is None:
        print("队列为空 / queue empty")
        return False
    job_id = job["id"]
    claim_token = job.get("claim_token", "")
    claim_headers = {"X-Claim-Token": claim_token}
    work_dir = os.path.join(WORK_ROOT, job_id)
    video_path = os.path.join(work_dir, "input.mp4")
    out_dir = os.path.join(work_dir, "outputs")
    print("领到任务 %s / claimed job" % job_id)
    result_posted = False
    try:
        # makedirs 挪进 try:失败(如磁盘只读/权限)也要走 /fail,不然任务卡在 processing
        # 直到 24h 自愈,白等一轮。
        # makedirs moved inside try: a failure here (e.g. read-only disk / permissions)
        # must also go through /fail, otherwise the job sits stuck until the 24h self-heal.
        os.makedirs(work_dir, exist_ok=True)
        _download(client, job["video_url"], video_path)
        timeout_sec = pipeline_timeout(job["duration_sec"])
        try:
            proc = subprocess.run(
                build_cli_args(video_path, out_dir, job.get("params")),
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            # pipeline 跑超时视为 pipeline_error,在 result POST 之前发生,/fail 合适。
            # timeout is treated as pipeline_error; happens before result POST, /fail is correct.
            raise RuntimeError("pipeline timeout after %ds" % timeout_sec)
        if proc.returncode != 0:
            raise RuntimeError("pipeline exit %d" % proc.returncode)
        report = build_report(out_dir)
        hl = has_highlights(out_dir)

        resp = with_retry(
            lambda: client.post(
                "/v1/rallyvision/worker/jobs/%s/result" % job_id,
                json={"report": report, "has_highlights": hl},
                headers=claim_headers,
            ).raise_for_status()
        ).json()
        # result 一旦提交成功,后面任何失败都不能再 /fail——数据已经在后端落库,/fail 会把
        # 这轮真实成果错误标成 pipeline_error 且 24h 自愈只救 processing 状态救不回来。
        # Once result POST succeeds, no later failure may call /fail — the data is already
        # persisted; /fail would mislabel real success as pipeline_error, and the 24h
        # self-heal only rescues `processing`, not `failed`.
        result_posted = True

        if hl:
            put_url = resp.get("highlights_put_url")
            if not put_url:
                raise RuntimeError(
                    "has_highlights=true 但后端未返回 highlights_put_url / "
                    "backend returned no highlights_put_url despite has_highlights=true"
                )
            with open(os.path.join(out_dir, "highlights.mp4"), "rb") as f:
                highlight_bytes = f.read()
            with_retry(
                lambda: httpx.put(
                    put_url, content=highlight_bytes,
                    headers={"Content-Type": "video/mp4"}, timeout=600.0,
                ).raise_for_status()
            )

        with_retry(
            lambda: client.post(
                "/v1/rallyvision/worker/jobs/%s/complete" % job_id, headers=claim_headers
            ).raise_for_status()
        )
        print("任务完成 %s / job done" % job_id)
    except Exception as exc:  # noqa: BLE001 — 任何失败都要上报或至少记日志,不能吞
        if result_posted:
            print(
                "任务 %s: result 已提交,后续步骤失败(%s)但不再 /fail,留给 24h 自愈重新处理 / "
                "job %s: result already submitted, later step failed (%s) — skipping /fail, "
                "leaving job for 24h self-heal to re-queue" % (job_id, exc, job_id, exc),
                file=sys.stderr,
            )
            return True
        code = pick_error_code(exc)
        print("任务失败 %s: %s (%s) / job failed" % (job_id, exc, code), file=sys.stderr)
        try:
            client.post("/v1/rallyvision/worker/jobs/%s/fail" % job_id,
                        json={"error_code": code}, headers=claim_headers).raise_for_status()
        except Exception as report_exc:  # 上报失败只打日志,等 24h 自愈重置
            print("上报失败 %s / fail-report failed" % report_exc, file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    client = _client()
    if args.loop:
        while True:
            try:
                process_one(client)
            except Exception as exc:  # claim 本身失败(网络/后端挂)也不能杀循环
                print("轮询异常 %s / poll error" % exc, file=sys.stderr)
            time.sleep(args.interval)
    else:
        process_one(client)


if __name__ == "__main__":
    main()
