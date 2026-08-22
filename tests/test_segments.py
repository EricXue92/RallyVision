from tennis_analysis.analysis.segments import extract_segments


def _mk_points():
    # 120 帧：0-39 球从 y=21 飞向 y=4（lower 击球），40 弹跳，41-79 飞回（upper 击球），80 弹跳...
    pts = []
    for f in range(120):
        if f <= 40: y = 21 - (17 * f / 40)
        elif f <= 80: y = 4 + (17 * (f - 40) / 40)
        else: y = 21 - (17 * (f - 80) / 40)
        pts.append({"frame": f, "time_sec": f / 60.0, "image": [640, 360], "court": [5.5, y]})
    return pts


BOUNCES = [{"frame": 40, "court": [5.5, 4.2]}, {"frame": 80, "court": [5.5, 20.8]}]
PLAYERS = {"upper": [5.5, 2.0], "lower": [5.5, 21.0]}


def test_extracts_two_complete_segments():
    segs = extract_segments(_mk_points(), BOUNCES, PLAYERS, fps=60.0)
    assert len(segs) == 2
    assert segs[0].hitter == "lower" and segs[0].bounce_frame == 40
    assert segs[1].hitter == "upper" and segs[1].bounce_frame == 80
    assert segs[0].hit_frame <= 5      # 起点在方向确立的最初几帧


def test_segment_too_short_dropped():
    segs = extract_segments(_mk_points()[:20], [], PLAYERS, fps=60.0)
    assert segs == []                  # 无弹跳事件配对的段丢弃
