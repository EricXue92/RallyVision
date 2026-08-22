"""段级到 shot_metrics.json 的纯后处理链编排（Task 10 binding，见 task-10-brief Step 6）。

顺序：extract_segments -> compute_shot_metrics（camera=None 时降级，只留 line_call）
     -> classify_spin（仅 fit_ok 段）-> call_bounce（line_call_mode=None/"off" 时跳过）。

system.py `_cleanup` 与 tests/test_outputs_schema.py 必须调用**同一份** `compute_shot_metrics_entries`
——不许各自抄一份链，否则测试验证的不是生产代码实际路径。
"""
import math

from .line_call import call_bounce
from .segments import extract_segments
from .shot_metrics import compute_shot_metrics
from .spin import classify_spin
from ..data.writer import write_json

# 弹跳前后各取多少帧计算 image 空间竖直速度（喂给 classify_spin 的 pre/post_bounce_vy）。
BOUNCE_VY_WINDOW_FRAMES = 5


def compute_shot_metrics_entries(points, bounce_events, player_positions, fps, camera, line_call_mode="doubles"):
    """单个连续弹道片段（通常=一个回合）的纯后处理链，产出 shot_metrics.json 条目列表。

    Args:
        points: 清洗后轨迹点 dict 列表（cleaned_ball_trajectory.json 的 points schema：
            frame/time_sec/image/court）。
        bounce_events: 弹跳事件 dict 列表（frame/court）。
        player_positions: {"upper": [x, y], "lower": [x, y]}（court 坐标，米）。
        fps: 帧率。
        camera: CameraModel 或 None（None = homography-only 降级，跳过 speed/spin 拟合）。
        line_call_mode: "singles" | "doubles" 正常判罚；None 或 "off" 跳过 call_bounce，
            此时每条 line_call 恒为 None（对应 --line-call off）。

    Returns:
        list[dict]，每条含 compute_shot_metrics 的 7 个 key 加 spin_label/spin_confidence/
        line_call（binding 完整 schema，见 task-10-brief）。rms_px 的 inf 在此已转成 None
        （JSON 无 Infinity 字面量，下游/overlay 按 None 处理成 "—"）。
    """
    segments = extract_segments(points, bounce_events, player_positions, fps)
    if not segments:
        return []

    if camera is not None:
        raw_metrics = compute_shot_metrics(segments, camera)
    else:
        print(
            "提示：相机未标定（homography-only 降级），跳过速度/旋转拟合，仅保留 line_call / "
            "Notice: no calibrated camera (homography-only degrade), skipping speed/spin fit, "
            "keeping line_call only"
        )
        raw_metrics = [_degraded_metric(segment) for segment in segments]

    points_by_frame = {int(point["frame"]): point for point in points}
    skip_line_call = line_call_mode in (None, "off")

    entries = []
    for segment, metric in zip(segments, raw_metrics):
        entry = dict(metric)
        entry["rms_px"] = _sanitize_rms(entry.get("rms_px"))
        entry["spin_label"], entry["spin_confidence"] = _spin_for_entry(entry, segment, points_by_frame, fps)
        entry["line_call"] = None
        if not skip_line_call:
            verdict, _distance = call_bounce(segment.bounce_court_xy, mode=line_call_mode)
            entry["line_call"] = verdict
        entries.append(entry)
    return entries


def write_shot_metrics(path, entries):
    """把 compute_shot_metrics_entries 的结果写到 outputs/<video>/shot_metrics.json。"""
    write_json(path, entries)


def _degraded_metric(segment):
    return {
        "hit_frame": segment.hit_frame,
        "bounce_frame": segment.bounce_frame,
        "hitter": segment.hitter,
        "speed_kmh": None,
        "spin_coeff": None,
        "fit_ok": False,
        "rms_px": None,
    }


def _sanitize_rms(value):
    """rms_px 的 None/inf 统一转 None——json.dump 默认会把 inf 写成非法 JSON 的 Infinity 字面量。"""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value


def _spin_for_entry(entry, segment, points_by_frame, fps):
    """classify_spin 仅对 fit_ok 段调用（binding：拟合失败连 spin_coeff 都没有，没法判定）。"""
    if not entry.get("fit_ok"):
        return None, None

    velocities = _bounce_vertical_velocities(points_by_frame, segment.bounce_frame, fps)
    if velocities is None:
        print(
            f"警告：第 {segment.bounce_frame} 帧弹跳附近轨迹点不足，跳过旋转判定 / "
            f"Warning: insufficient trajectory points around bounce frame {segment.bounce_frame}, "
            "skipping spin classification"
        )
        return None, None

    pre_vy, post_vy = velocities
    try:
        # wrist_dy=None：姿态腕点数据暂未接线（noted future enhancement，见 task-10-brief）。
        return classify_spin(entry["spin_coeff"], pre_vy, post_vy, wrist_dy=None)
    except Exception as exc:  # noqa: BLE001 - 旋转判定失败不能中断整条流水线
        print(f"警告：旋转判定异常，已跳过 / Warning: spin classification raised, skipping: {exc}")
        return None, None


def _bounce_vertical_velocities(points_by_frame, bounce_frame, fps, window=BOUNCE_VY_WINDOW_FRAMES):
    pre = _velocity_in_range(points_by_frame, bounce_frame - window, bounce_frame, fps)
    post = _velocity_in_range(points_by_frame, bounce_frame, bounce_frame + window, fps)
    if pre is None or post is None:
        return None
    return pre, post


def _velocity_in_range(points_by_frame, start_frame, end_frame, fps):
    """区间内首尾两个有效 image 点的平均竖直速度（px/s，y-down 正）；点不足 2 个则不可算。"""
    ys = []
    for frame in range(start_frame, end_frame + 1):
        point = points_by_frame.get(frame)
        if point is None or point.get("image") is None:
            continue
        ys.append((frame, float(point["image"][1])))
    if len(ys) < 2:
        return None
    (f0, y0), (f1, y1) = ys[0], ys[-1]
    dt = (f1 - f0) / float(fps)
    if dt <= 0:
        return None
    return (y1 - y0) / dt
