"""真实数据球速验证工具 / Real-data serve-speed validation tool.

Task 11：对一批「带球速字幕的真实转播发球片段」跑完整 pipeline，取每条素材里
离 `hit_frame_approx` 最近的 `fit_ok=True` 段，算 `|speed_kmh - caption_kmh| / caption_kmh`
相对误差，打印表格与中位数——验收标准 2（真实数据球速误差中位数 < 10%）的执行工具。

Runs the full pipeline over a batch of real broadcast serve clips with speed
captions, picks the `fit_ok=True` segment nearest each clip's `hit_frame_approx`,
computes the relative speed error against the caption, and prints a table +
median — the tool that exercises acceptance criterion 2 (median real-data
speed error < 10%).

Manifest 格式 / manifest format（JSON 数组，每条一个素材）：

    [
      {"video": "videos/serve_01.mp4", "hit_frame_approx": 142, "caption_kmh": 187},
      {"video": "videos/serve_02.mp4", "hit_frame_approx": 88, "caption_kmh": 201, "label": "Alcaraz ace"}
    ]

字段 / fields：
    video            视频路径（相对项目根目录或绝对路径）/ video path (repo-relative or absolute)
    hit_frame_approx 目视估计的击球帧号（整数）/ eyeballed hit-frame index (int)
    caption_kmh      转播字幕给出的球速（正数）/ speed shown in the broadcast caption (positive number)
    label            可选，表格里显示的名称，默认回退到 video 路径 / optional display label, defaults to video path

用法 / usage：

    uv run tools/validate_speed.py --manifest tools/serve_manifest.json
    uv run tools/validate_speed.py --manifest tools/serve_manifest.json --ball-detector tracknet
    uv run tools/validate_speed.py --triage outputs/demo   # 只跑 Step 3 诊断，不跑 pipeline

真实的 ≥10 条转播素材（owner 手工准备，不入 git）应放进 `tools/serve_manifest.json`
（`.gitignore` 里 `tools/*` 默认忽略，只放行 `validate_speed.py` 和 `demo_manifest.json`）。
`tools/demo_manifest.json` 是仅用于验证工具链路的演示 manifest（1 条，指向仓库自带的
`videos/demo.mp4`），不是精度声明——demo.mp4 的已知问题见 README「已知精度状态」。

注意 / note：每条素材第一次跑 pipeline 时，若对应输出目录下没有缓存的
`court_annotations.txt`，`main.py` 的球场自动检测预览窗口会用 `cv2.waitKey(0)`
**阻塞等键盘输入**（Enter/Y 确认，或 M/R/Esc 转手动四角标注）——这是 pipeline 本身
一贯的行为（见 README「第一次运行流程」），本工具不新增/不绕过这个交互，只是按
视频文件名复用输出目录（`outputs/<video_stem>` 或 `--output-root` 下的同名子目录），
让同一支视频的重复验证跑不用每次都重新确认。首次对一批新素材跑验证前，建议先对每条
单独跑一遍 `uv run main.py --video-path <clip> ...` 走完交互确认，把 `court_annotations.txt`
缓存下来，再跑本工具做批量验证。

Each clip's first pipeline run will **block on `cv2.waitKey(0)`** waiting for a
keypress if its output directory has no cached `court_annotations.txt` yet — this
is existing `main.py` behavior (see the "First Run" section of the README), not
something this tool adds or bypasses. The tool keys output directories by video
filename stem so repeated validation runs on the same clip reuse the cached
annotation. Before batch-validating a fresh set of clips, it's best to run
`uv run main.py --video-path <clip> ...` once per clip interactively first so the
annotation gets cached.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEMPLATE_PATH = "templates/demo.png"
# 默认落在 main.py 本身的输出根目录（`outputs/`），让 `outputs/<video_stem>` 与
# main.py 单独跑该视频时的默认输出目录完全一致——同一支视频不管是手动跑 main.py
# 预热标定缓存，还是被本工具跑，都落在同一目录，天然复用 court_annotations.txt。
# 想隔离验证产物时用 --output-root 显式指定别的目录。
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs")
# 与 tennis_analysis.analysis.trajectory3d.fit_segment 的 max_rms 默认值保持一致，
# 仅用于诊断展示，不回写拟合器。
RMS_THRESHOLD_PX = 12.0

REQUIRED_MANIFEST_KEYS = ("video", "hit_frame_approx", "caption_kmh")


class ManifestError(ValueError):
    """manifest 格式错误 / invalid manifest format."""


# ---------------------------------------------------------------------------
# Manifest 解析 / manifest parsing
# ---------------------------------------------------------------------------

def load_manifest(path):
    """读取并校验 manifest JSON，返回标准化条目列表。

    Read and validate the manifest JSON, returning a list of normalized entries.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ManifestError(
            f"manifest 必须是非空 JSON 数组 / manifest must be a non-empty JSON array: {path}"
        )

    entries = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ManifestError(f"第 {i} 条不是对象 / entry {i} is not a JSON object")

        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in raw]
        if missing:
            raise ManifestError(f"第 {i} 条缺少字段 {missing} / entry {i} missing keys {missing}")

        video = raw["video"]
        hit_frame_approx = raw["hit_frame_approx"]
        caption_kmh = raw["caption_kmh"]

        if not isinstance(video, str) or not video:
            raise ManifestError(f"第 {i} 条 video 必须是非空字符串 / entry {i} video must be a non-empty string")
        if not isinstance(hit_frame_approx, int) or isinstance(hit_frame_approx, bool) or hit_frame_approx < 0:
            raise ManifestError(
                f"第 {i} 条 hit_frame_approx 必须是非负整数 / entry {i} hit_frame_approx must be a non-negative int"
            )
        if not isinstance(caption_kmh, (int, float)) or isinstance(caption_kmh, bool) or caption_kmh <= 0:
            raise ManifestError(
                f"第 {i} 条 caption_kmh 必须是正数 / entry {i} caption_kmh must be a positive number"
            )

        entries.append({
            "video": video,
            "hit_frame_approx": int(hit_frame_approx),
            "caption_kmh": float(caption_kmh),
            "label": raw.get("label") or video,
            "template": raw.get("template") or DEFAULT_TEMPLATE_PATH,
        })
    return entries


# ---------------------------------------------------------------------------
# 段选取 / 误差 / 中位数 — Step 2 核心纯逻辑
# ---------------------------------------------------------------------------

def select_nearest_fit_ok_segment(shot_metrics_entries, hit_frame_approx):
    """在 shot_metrics 条目里选出 `fit_ok=True` 且 `hit_frame` 距 `hit_frame_approx` 最近的一条。

    无任何 fit_ok 段时返回 None——调用方须友好处理（真实素材/已知问题下大概率会遇到），
    不抛异常。

    Pick the `fit_ok=True` entry whose `hit_frame` is closest to `hit_frame_approx`.
    Returns None when no entry has `fit_ok=True` — callers must handle this gracefully.
    """
    candidates = [e for e in shot_metrics_entries if e.get("fit_ok")]
    if not candidates:
        return None
    return min(candidates, key=lambda e: (abs(e["hit_frame"] - hit_frame_approx), e["hit_frame"]))


def compute_relative_error(speed_kmh, caption_kmh):
    """`|speed_kmh - caption_kmh| / caption_kmh`。"""
    if caption_kmh <= 0:
        raise ValueError("caption_kmh 必须 > 0 / caption_kmh must be > 0")
    return abs(speed_kmh - caption_kmh) / caption_kmh


def median_relative_error(errors):
    """errors 为空时返回 None（调用方展示"无有效样本"），否则返回中位数。"""
    if not errors:
        return None
    return statistics.median(errors)


# ---------------------------------------------------------------------------
# Pipeline 编排 / pipeline orchestration
# ---------------------------------------------------------------------------

def build_pipeline_args(video_path, template_path, output_dir, ball_detector="yolo", line_call="doubles"):
    """构造喂给 main.py 的 argv（不含 python/uv 前缀），纯函数，便于测试。

    main.py 目前没有帧窗口截取参数，只能整段跑；验证脚本不新增 pipeline 特性
    （见 task-11 brief 的范围约束），选段逻辑全靠 Step 2 的最近 fit_ok 段挑选。
    """
    return [
        "--video-path", video_path,
        "--template-path", template_path,
        "--output-dir", output_dir,
        "--ball-detector", ball_detector,
        "--shot-metrics", "true",
        "--line-call", line_call,
        "--display", "false",
        "--visualize-positions", "false",
        "--audio", "false",
    ]


def output_dir_for(video_path, root=None):
    """按视频文件名 stem 分配输出目录（不带序号）——同一支视频的重复验证跑复用同一目录，

    从而复用已缓存的 `court_annotations.txt`，避免每次都触发 main.py 的交互式球场确认
    （见模块docstring「注意」）。root 默认落在 `outputs/validate_speed/`；显式传 `root=None`
    以外的值时，多个 manifest 条目引用同一视频会天然去重到同一目录。

    Output directory keyed by video filename stem (no index) — repeated validation
    runs on the same clip reuse the same directory and its cached
    `court_annotations.txt`, avoiding main.py's interactive court-confirmation
    prompt on every rerun (see module docstring "note").
    """
    root = root or DEFAULT_OUTPUT_ROOT
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(root, stem)


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def run_pipeline(video_path, template_path, output_dir, ball_detector="yolo", line_call="doubles",
                  main_py=None, python_exe=None):
    """真跑一遍 main.py pipeline（子进程），返回 (returncode, stdout, stderr)。"""
    main_py = main_py or os.path.join(REPO_ROOT, "main.py")
    python_exe = python_exe or sys.executable
    args = build_pipeline_args(video_path, template_path, output_dir, ball_detector, line_call)
    cmd = [python_exe, main_py] + args
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def load_shot_metrics(output_dir):
    path = os.path.join(output_dir, "shot_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cleaned_trajectory(output_dir):
    path = os.path.join(output_dir, "cleaned_ball_trajectory.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("points", [])


# ---------------------------------------------------------------------------
# 诊断统计（Step 3：球检测缺帧率 / rms_px 分布）
# ---------------------------------------------------------------------------

def _gap_run_lengths(gap_flags):
    """把 bool 序列的连续 True 段编码成游程长度列表。"""
    runs = []
    cur = 0
    for g in gap_flags:
        if g:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def trajectory_visibility_stats(points):
    """整段清洗轨迹的球检测可见帧统计。

    可见 = 有原始检测且未插值；gap（缺帧）= 原始缺测（image=None）或插值填补。
    用于诊断①「球检测缺帧率」是不是 fit_ok=False 的主因。

    Whole-clip visibility stats for `cleaned_ball_trajectory.json`. "Visible" means
    a raw (non-interpolated) detection exists; a "gap" is either a raw miss or an
    interpolated fill-in. Used to diagnose whether ball-detection dropout explains
    fit_ok=False.
    """
    ordered = sorted(points, key=lambda p: p["frame"])
    total = len(ordered)
    gap_flags = [(p.get("image") is None) or bool(p.get("interpolated")) for p in ordered]
    visible = total - sum(gap_flags)
    raw_missing = sum(1 for p in ordered if p.get("image") is None)
    interpolated = sum(1 for p in ordered if p.get("interpolated"))
    runs = _gap_run_lengths(gap_flags)
    return {
        "total_frames": total,
        "visible_frames": visible,
        "visible_pct": round(100.0 * visible / total, 2) if total else 0.0,
        "raw_missing_frames": raw_missing,
        "interpolated_frames": interpolated,
        "gap_runs": runs,
        "max_gap_run": max(runs) if runs else 0,
        "median_gap_run": statistics.median(runs) if runs else 0,
    }


def segment_visibility_stats(points, hit_frame, bounce_frame):
    """单个击球段 `[hit_frame, bounce_frame]`（闭区间）内的球检测缺帧统计。

    轨迹里完全没有该帧记录，和该帧记录存在但 `image=None`，都算 raw_missing——
    对拟合器来说两者都是"这一帧没有像素观测"。

    Per-segment visibility stats over the closed interval `[hit_frame, bounce_frame]`.
    A frame absent from the trajectory entirely counts the same as a frame present
    with `image=None` — both mean "no pixel observation this frame" to the fitter.
    """
    by_frame = {p["frame"]: p for p in points}
    frames = list(range(hit_frame, bounce_frame + 1))
    gap_flags = []
    raw_missing = 0
    for f in frames:
        p = by_frame.get(f)
        if p is None or p.get("image") is None:
            raw_missing += 1
            gap_flags.append(True)
        else:
            gap_flags.append(bool(p.get("interpolated")))
    runs = _gap_run_lengths(gap_flags)
    n = len(frames)
    gap_count = sum(gap_flags)
    return {
        "hit_frame": hit_frame,
        "bounce_frame": bounce_frame,
        "n_frames": n,
        "gap_frames": gap_count,
        "gap_pct": round(100.0 * gap_count / n, 2) if n else 0.0,
        "raw_missing_frames": raw_missing,
        "max_gap_run": max(runs) if runs else 0,
    }


def rms_px_distribution(shot_metrics_entries, threshold=RMS_THRESHOLD_PX):
    """`rms_px` 分布统计（只统计非 None 值），含超阈值计数。用于诊断③。"""
    values = [e["rms_px"] for e in shot_metrics_entries if e.get("rms_px") is not None]
    fit_ok_count = sum(1 for e in shot_metrics_entries if e.get("fit_ok"))
    fit_fail_count = len(shot_metrics_entries) - fit_ok_count
    if not values:
        return {
            "count": 0, "min": None, "median": None, "max": None,
            "over_threshold": 0, "fit_ok_count": fit_ok_count, "fit_fail_count": fit_fail_count,
        }
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "over_threshold": sum(1 for v in values if v >= threshold),
        "fit_ok_count": fit_ok_count,
        "fit_fail_count": fit_fail_count,
    }


# ---------------------------------------------------------------------------
# 编排入口 / orchestration entry points
# ---------------------------------------------------------------------------

def validate(manifest_path, ball_detector="yolo", line_call="doubles", output_root=None, skip_pipeline=False):
    """跑完 manifest 里每条素材，返回逐条结果行（含误差或跳过原因）。"""
    entries = load_manifest(manifest_path)
    rows = []
    for item in entries:
        out_dir = output_dir_for(item["video"], output_root)
        video_path = _resolve(item["video"])
        template_path = _resolve(item["template"])

        shot_metrics_path = os.path.join(out_dir, "shot_metrics.json")
        if not (skip_pipeline and os.path.exists(shot_metrics_path)):
            rc, out, err = run_pipeline(video_path, template_path, out_dir, ball_detector, line_call)
            if rc != 0:
                rows.append({**item, "output_dir": out_dir, "status": f"pipeline exit={rc}"})
                continue

        sm = load_shot_metrics(out_dir)
        if not sm:
            rows.append({**item, "output_dir": out_dir, "status": "no shot_metrics.json"})
            continue

        seg = select_nearest_fit_ok_segment(sm, item["hit_frame_approx"])
        if seg is None:
            rows.append({**item, "output_dir": out_dir, "status": "no fit_ok segment"})
            continue

        error = compute_relative_error(seg["speed_kmh"], item["caption_kmh"])
        rows.append({
            **item,
            "output_dir": out_dir,
            "status": "ok",
            "matched_hit_frame": seg["hit_frame"],
            "speed_kmh": seg["speed_kmh"],
            "error": error,
        })
    return rows


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def print_report(rows):
    """打印逐条误差表格 + 中位数（bilingual）。"""
    print("\n=== 真实数据球速验证结果 / Real-data speed validation results ===")
    header = f"{'label':<28}{'hit≈':>6}{'matched':>9}{'caption':>9}{'speed':>9}{'error':>9}  status"
    print(header)
    print("-" * len(header))

    errors = []
    for r in rows:
        label = r["label"][:27]
        hit_approx = r["hit_frame_approx"]
        if r.get("status") == "ok":
            errors.append(r["error"])
            print(f"{label:<28}{hit_approx:>6}{r['matched_hit_frame']:>9}"
                  f"{r['caption_kmh']:>9.1f}{r['speed_kmh']:>9.1f}{_fmt_pct(r['error']):>9}  ok")
        else:
            print(f"{label:<28}{hit_approx:>6}{'—':>9}{r['caption_kmh']:>9.1f}{'—':>9}{'—':>9}  {r['status']}")

    print("-" * len(header))
    median = median_relative_error(errors)
    if median is None:
        print("中位数误差：无有效样本（全部素材都没有 fit_ok 段）/ "
              "Median error: no valid samples (no clip produced a fit_ok segment)")
        print("→ 参见 README「已知精度状态」了解已知的真实数据拟合问题。/ "
              "See README's known accuracy status section for the known real-data fitting issue.")
    else:
        print(f"中位数误差 / Median error: {_fmt_pct(median)}  (n={len(errors)}/{len(rows)})")
        if median >= 0.10:
            print("中位数 ≥ 10%，按 brief Step 3 顺序排查：①标定重投影误差 ②球检测缺帧率 ③拟合 rms_px 分布。/ "
                  "Median >= 10%: triage in order per the brief's Step 3 — "
                  "(1) calibration reprojection error, (2) ball-detection miss rate, (3) fit rms_px distribution.")
            print("可用 --triage <output_dir> 跑②③两项诊断。/ "
                  "Use --triage <output_dir> to run diagnostics (2) and (3).")
    return median


def run_triage(output_dir):
    """Step 3 诊断：整段可见率/缺帧游程 + 逐段缺帧率 + rms_px 分布，不修拟合器，只打印。"""
    sm = load_shot_metrics(output_dir)
    traj = load_cleaned_trajectory(output_dir)
    if sm is None or traj is None:
        print(f"缺少 shot_metrics.json 或 cleaned_ball_trajectory.json / "
              f"missing shot_metrics.json or cleaned_ball_trajectory.json under: {output_dir}")
        return 1

    print(f"\n=== Step 3 诊断 / triage: {output_dir} ===")

    vis = trajectory_visibility_stats(traj)
    print(f"\n② 整段球检测可见率 / whole-clip ball-detection visibility:")
    print(f"  total_frames={vis['total_frames']}  visible={vis['visible_frames']} "
          f"({vis['visible_pct']:.1f}%)  raw_missing={vis['raw_missing_frames']}  "
          f"interpolated={vis['interpolated_frames']}")
    print(f"  gap_runs={vis['gap_runs']}  max_gap_run={vis['max_gap_run']}  "
          f"median_gap_run={vis['median_gap_run']}")

    print(f"\n  逐段（hit→bounce）缺帧率 / per-segment gap rate:")
    for e in sm:
        seg_vis = segment_visibility_stats(traj, e["hit_frame"], e["bounce_frame"])
        rms_str = "—" if e.get("rms_px") is None else f"{e['rms_px']:.2f}"
        print(f"    hit={e['hit_frame']:>4} bounce={e['bounce_frame']:>4} "
              f"n={seg_vis['n_frames']:>3} gap={seg_vis['gap_frames']:>2}({seg_vis['gap_pct']:>5.1f}%) "
              f"max_run={seg_vis['max_gap_run']} fit_ok={e.get('fit_ok')} rms_px={rms_str}")

    rms_stats = rms_px_distribution(sm)
    print(f"\n③ rms_px 分布 / rms_px distribution (threshold={RMS_THRESHOLD_PX}px):")
    if rms_stats["count"] == 0:
        print("  无可用 rms_px 值 / no rms_px values available")
    else:
        print(f"  count={rms_stats['count']}  min={rms_stats['min']:.2f}  "
              f"median={rms_stats['median']:.2f}  max={rms_stats['max']:.2f}  "
              f"over_threshold={rms_stats['over_threshold']}/{rms_stats['count']}")
    print(f"  fit_ok={rms_stats['fit_ok_count']}  fit_fail={rms_stats['fit_fail_count']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="真实数据球速验证工具 / real-data speed validation tool（Task 11）")
    parser.add_argument("--manifest", default=None,
                         help="manifest JSON 路径 / path to manifest JSON")
    parser.add_argument("--ball-detector", default="yolo", choices=["yolo", "tracknet", "wasb"],
                         help="球检测后端 / ball detector backend")
    parser.add_argument("--line-call", default="doubles", choices=["singles", "doubles", "off"],
                         help="落点判罚场地模式 / line-call court mode")
    parser.add_argument("--output-root", default=None,
                         help="输出根目录，默认 outputs/validate_speed / output root, default outputs/validate_speed")
    parser.add_argument("--skip-pipeline", action="store_true",
                         help="目标输出目录已有 shot_metrics.json 时复用、不重跑 pipeline / "
                              "reuse existing shot_metrics.json instead of rerunning the pipeline")
    parser.add_argument("--triage", default=None, metavar="OUTPUT_DIR",
                         help="对已有输出目录跑 Step3 诊断（不跑 pipeline）/ "
                              "run Step-3 diagnostics against an existing output dir (no pipeline run)")
    args = parser.parse_args(argv)

    if args.triage:
        return run_triage(args.triage)

    if not args.manifest:
        parser.error("需要 --manifest 或 --triage 之一 / need either --manifest or --triage")

    rows = validate(args.manifest, args.ball_detector, args.line_call, args.output_root, args.skip_pipeline)
    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
