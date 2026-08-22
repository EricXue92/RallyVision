"""IN/OUT 判罚：压线即界内规则的落点边界判定。

court_xy 坐标系与 court/mapper.py 一致：x∈[0,W] 场宽方向，y∈[0,23.77] 场长方向。
"""
import math

from tennis_analysis.analysis.physics import BALL_RADIUS
from tennis_analysis.court.mapper import (
    TENNIS_COURT_LENGTH,
    TENNIS_DOUBLES_WIDTH,
    TENNIS_SINGLES_WIDTH,
)

_SINGLES_MARGIN = (TENNIS_DOUBLES_WIDTH - TENNIS_SINGLES_WIDTH) / 2


def call_bounce(court_xy, mode="doubles", close_margin_m=0.15):
    """判定落点相对场地边界的 in/out/close，压线（含 close 带）算界内友好判罚。

    返回 (verdict, distance_to_line_m)：verdict ∈ {"in","out","close"}，
    distance_to_line_m 是到最近边界线的距离（>=0）。
    """
    x, y = court_xy
    if mode == "singles":
        xmin, xmax = _SINGLES_MARGIN, TENNIS_DOUBLES_WIDTH - _SINGLES_MARGIN
    elif mode == "doubles":
        xmin, xmax = 0.0, TENNIS_DOUBLES_WIDTH
    else:
        raise ValueError(f"unknown mode: {mode!r}, expected 'singles' or 'doubles'")
    ymin, ymax = 0.0, TENNIS_COURT_LENGTH

    # 矩形边界的有符号距离场（SDF）：负=界内（|值|=到最近线的距离），正=界外（=到矩形最近点的欧氏距离）
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    hx, hy = (xmax - xmin) / 2, (ymax - ymin) / 2
    qx = abs(x - cx) - hx
    qy = abs(y - cy) - hy
    outside_dist = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside_dist = min(max(qx, qy), 0.0)
    signed_distance = outside_dist + inside_dist

    distance_to_line_m = abs(signed_distance)

    if distance_to_line_m < close_margin_m:
        verdict = "close"
    elif signed_distance > BALL_RADIUS:
        # 整球出界规则：球只要有任何部分触线即算压线（界内友好判罚）。
        # 球心到线距离 < 半径 → 球体与线仍有接触 → 界内；> 半径才是整球出界。
        # （此前误用直径，R16 判罚裁定为半径口径）
        verdict = "out"
    else:
        verdict = "in"

    return verdict, distance_to_line_m
