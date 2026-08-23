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
    work_dir = os.path.join(WORK_ROOT, job_id)
    os.makedirs(work_dir, exist_ok=True)
    video_path = os.path.join(work_dir, "input.mp4")
    out_dir = os.path.join(work_dir, "outputs")
    print("领到任务 %s / claimed job" % job_id)
    try:
        _download(client, job["video_url"], video_path)
        proc = subprocess.run(build_cli_args(video_path, out_dir, job.get("params")),
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if proc.returncode != 0:
            raise RuntimeError("pipeline exit %d" % proc.returncode)
        report = build_report(out_dir)
        hl = has_highlights(out_dir)
        resp = client.post("/v1/rallyvision/worker/jobs/%s/result" % job_id,
                           json={"report": report, "has_highlights": hl}).raise_for_status().json()
        if hl and resp.get("highlights_put_url"):
            with open(os.path.join(out_dir, "highlights.mp4"), "rb") as f:
                up = httpx.put(resp["highlights_put_url"], content=f.read(),
                               headers={"Content-Type": "video/mp4"}, timeout=600.0)
                up.raise_for_status()
        client.post("/v1/rallyvision/worker/jobs/%s/complete" % job_id).raise_for_status()
        print("任务完成 %s / job done" % job_id)
    except Exception as exc:  # noqa: BLE001 — 任何失败都要上报,不能吞
        code = pick_error_code(exc)
        print("任务失败 %s: %s (%s) / job failed" % (job_id, exc, code), file=sys.stderr)
        try:
            client.post("/v1/rallyvision/worker/jobs/%s/fail" % job_id,
                        json={"error_code": code}).raise_for_status()
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
