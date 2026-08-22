from tennis_analysis.analysis.segments import extract_segments, refine_bounce


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


def test_hitter_survives_missing_court_at_reversal_frame():
    # Finding 1 回归：翻转帧（frame 40）原始 court 缺测，不能让 NaN 比较静默判成
    # "lower"——该帧真实半场是 upper（y≈4 < 11.885），须靠滑动平均降级判定。
    pts = _mk_points()
    pts[40]["court"] = None
    segs = extract_segments(pts, BOUNCES, PLAYERS, fps=60.0)
    assert len(segs) == 2
    assert segs[1].hitter == "upper"


def test_transient_glitch_does_not_corrupt_segment():
    # Finding 2 回归（controller ruling R9）：飞行中段 2-3 帧的尖锐抖动（frame 15-17）
    # 会在无滞回时造出 [0, 12, 14, 40, 80] 候选击球帧，导致第一段的 hit_frame 被
    # 抢帧为 14（应为 ~0），静默腐蚀真实分段。加了 MIN_REVERSAL_PERSIST_FRAMES=3
    # 滞回后，抖动应被过滤，两段的 hit_frame/bounce_frame/hitter 都应与干净轨迹一致。
    pts = _mk_points()
    for frame, glitch_dy in [(15, 3.0), (16, 4.0), (17, 1.0)]:
        x, y = pts[frame]["court"]
        pts[frame]["court"] = [x, y + glitch_dy]

    segs = extract_segments(pts, BOUNCES, PLAYERS, fps=60.0)
    assert len(segs) == 2
    assert segs[0].hit_frame <= 5 and segs[0].bounce_frame == 40 and segs[0].hitter == "lower"
    assert segs[1].hit_frame == 40 and segs[1].bounce_frame == 80 and segs[1].hitter == "upper"


def test_short_segment_with_bounce_dropped():
    # 可选覆盖：命中「找到弹跳但段长 < 8 帧」分支（与「无弹跳」分支路径不同）。
    pts = [
        {"frame": f, "time_sec": f / 60.0, "image": [640, 360], "court": [5.5, 21.0 - 2.0 * f]}
        for f in range(10)
    ]
    bounce = [{"frame": 5, "court": [5.5, 21.0 - 2.0 * 5]}]
    segs = extract_segments(pts, bounce, PLAYERS, fps=60.0)
    assert segs == []                  # 弹跳配对成功但段长 5 < MIN_SEGMENT_FRAMES(8)，仍丢弃


def _v_shape_points(vertex=40.3):
    # image y 先降后升（V 形，谷底在 40.3 帧，亚帧），court y 全程匀速推进
    pts = []
    for f in range(25, 56):
        img_y = 500 + (12 * (vertex - f) if f < vertex else 9 * (f - vertex))
        pts.append({"frame": f, "time_sec": f / 60.0,
                    "image": [640.0, img_y], "court": [5.5, 21.0 - 0.42 * (f - 25)]})
    return pts


def test_refine_bounce_subframe_vertex():
    t_sub, court_xy = refine_bounce(_v_shape_points(), bounce_frame=40)
    assert abs(t_sub - 40.3) < 0.2
    assert abs(court_xy[1] - (21.0 - 0.42 * (40.3 - 25))) < 0.1   # court 按亚帧时刻插值


def test_refine_bounce_falls_back_when_sparse():
    pts = _v_shape_points()[13:18]          # 前后凑不齐 3 帧
    t_sub, court_xy = refine_bounce(pts, bounce_frame=40)
    assert t_sub == 40.0                     # 原样返回，不抛异常
