"""击球分段提取：从清洗后的球轨迹 + 弹跳事件切出 hit -> bounce 的逐拍片段。

算法（binding，见 task-5-brief Step 3）：
1. 对 court y 序列做 5 帧滑动平均，取符号变化帧作为候选击球帧（速度方向翻转）。
2. 轨迹起点本身也是候选击球帧（第一拍没有「前一次翻转」，方向一确立即算）。
3. 每个候选击球帧向后找「下一个候选击球帧之前」最近的弹跳事件配对成一段；
   段长 < 8 帧，或找不到可配对的弹跳事件，整段丢弃。
4. 击球者 = 翻转（候选击球）帧球所在半场（y < 11.885 -> upper，否则 lower）。
5. hit_hint_xyz = [击球者球员 court x, court y, 1.0]（球员位置来自 player_positions，
   非球的坐标；暂无姿态腕点数据，z 恒 1.0——后续如接入腕点可替换，见 Interfaces 注释）。
"""
from dataclasses import dataclass

import numpy as np

NET_Y = 11.885
MIN_SEGMENT_FRAMES = 8
SMOOTH_WINDOW = 5


@dataclass
class ShotSegment:
    hit_frame: int
    bounce_frame: int
    hitter: str              # 'upper' | 'lower'
    frame_times: list        # 相对 hit 的秒
    ball_px: list             # [N,2]，缺测为 None
    bounce_court_xy: list
    hit_hint_xyz: list       # [球员x, 球员y, 1.0]


def extract_segments(points, bounce_events, player_positions, fps) -> list:
    points = sorted(points, key=lambda p: p["frame"])
    if len(points) < 2:
        return []

    frames = [int(p["frame"]) for p in points]
    y = np.array(
        [p["court"][1] if p.get("court") is not None else np.nan for p in points],
        dtype=float,
    )
    smoothed = _moving_average(y, SMOOTH_WINDOW)
    velocity_sign = np.sign(np.diff(smoothed))

    hit_indices = _find_candidate_hits(velocity_sign)
    if not hit_indices:
        return []

    bounces_sorted = sorted(bounce_events, key=lambda event: event["frame"])
    end_sentinel = frames[-1] + 1

    segments = []
    for position, hit_idx in enumerate(hit_indices):
        hit_frame = frames[hit_idx]
        next_hit_frame = frames[hit_indices[position + 1]] if position + 1 < len(hit_indices) else end_sentinel

        bounce = _nearest_bounce(bounces_sorted, hit_frame, next_hit_frame)
        if bounce is None:
            continue
        bounce_frame = int(bounce["frame"])
        if bounce_frame - hit_frame < MIN_SEGMENT_FRAMES:
            continue

        hit_y = y[hit_idx]
        hitter = "upper" if hit_y < NET_Y else "lower"
        hitter_pos = player_positions[hitter]

        seg_points = [p for p in points if hit_frame <= p["frame"] <= bounce_frame]
        hit_time = points[hit_idx]["time_sec"]

        segments.append(
            ShotSegment(
                hit_frame=int(hit_frame),
                bounce_frame=bounce_frame,
                hitter=hitter,
                frame_times=[round(float(p["time_sec"]) - float(hit_time), 6) for p in seg_points],
                ball_px=[list(p["image"]) if p.get("image") is not None else None for p in seg_points],
                bounce_court_xy=list(bounce["court"]),
                hit_hint_xyz=[float(hitter_pos[0]), float(hitter_pos[1]), 1.0],
            )
        )

    return segments


def _moving_average(values, window):
    half = window // 2
    n = len(values)
    smoothed = np.full(n, np.nan, dtype=float)
    for index in range(n):
        lo = max(0, index - half)
        hi = min(n, index + half + 1)
        chunk = values[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            smoothed[index] = float(np.mean(valid))
    return smoothed


def _find_candidate_hits(velocity_sign):
    """候选击球帧 = 轨迹起点（方向一确立）+ 每次符号翻转的帧。"""
    candidates = []
    last_sign = None
    for index, sign in enumerate(velocity_sign):
        if sign == 0 or np.isnan(sign):
            continue
        if last_sign is None:
            candidates.append(index)  # 起点：方向确立即为候选击球帧
            last_sign = sign
            continue
        if sign != last_sign:
            candidates.append(index)  # 翻转帧
            last_sign = sign
    return candidates


def _nearest_bounce(bounces, hit_frame, next_hit_frame):
    candidates = [
        event for event in bounces
        if hit_frame < int(event["frame"]) <= next_hit_frame
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda event: int(event["frame"]))
