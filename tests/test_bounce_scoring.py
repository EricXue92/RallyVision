"""落点图事故回归(job 6e98cc64):弹跳事件混入过网点/球拍击球点,单帧 homography
把空中球投到地面,顶视图落点偏差数米(标注视频里像素标记贴着球走所以看不出)。

三条守护:
1. 图像 y 局部谷(屏幕最高点=过网/弧顶)物理上不可能是触地,必须拒绝。
2. court 深度方向(y)速度在中心帧反转 = 球拍击球,不是弹跳(真弹跳后球继续
   朝同一深度方向走)。
3. 事件帧吸附到邻域图像 y 最大帧 + 双抛物线亚帧插值 court,替换单帧投影。
"""
import numpy as np

from tennis_analysis.analysis.bounce import BounceDetector, TrajectoryPoint


def _detector():
    return BounceDetector(fps=25.0)


def _velocity(detector, coords):
    return detector._velocity(np.asarray(coords, dtype=np.float32))


# 与生产调用一致:window_size=20, center_offset=10 -> center = 20 - 10 - 1 = 9
CENTER = 9


def _window(xs, ys):
    return np.asarray(list(zip(xs, ys)), dtype=np.float32)


def test_screen_y_valley_apex_is_rejected():
    """过网/弧顶:球在屏幕上先升(y 减)后降(y 增),中心是局部 y 谷。
    job 6e98cc64 f16 就是这样的过网点,当前被记成 confidence 0.633 的弹跳。"""
    detector = _detector()
    xs = [300 + 12 * i for i in range(20)]
    ys = [400 - 10 * i if i <= CENTER else 400 - 10 * CENTER + 10 * (i - CENTER)
          for i in range(20)]
    window = _window(xs, ys)
    velocity = _velocity(detector, window)

    score, diagnostics = detector._score_window(window, velocity, CENTER)

    assert score == 0.0
    assert diagnostics.get("reject_reason") == "no_ground_contact_signature"


def test_court_depth_reversal_racket_contact_is_rejected():
    """球拍击球:图像 y 有局部峰(球落向球拍再被打回),但 court 深度方向速度
    在中心帧反转。job 6e98cc64 f142/f217 都是击球点被当弹跳。"""
    detector = _detector()
    xs = [300 + 10 * i for i in range(20)]
    ys = [400 + 10 * i if i <= CENTER else 400 + 10 * CENTER - 10 * (i - CENTER)
          for i in range(20)]
    window = _window(xs, ys)
    velocity = _velocity(detector, window)
    # 球飞向近端球员(court y 增),被击回(court y 减),x 不变
    cy = [15.0 + 0.4 * i if i <= CENTER else 15.0 + 0.4 * CENTER - 0.4 * (i - CENTER)
          for i in range(20)]
    court_window = _window([5.0] * 20, cy)

    score, diagnostics = detector._score_window(window, velocity, CENTER, court_window=court_window)

    assert score == 0.0
    assert diagnostics.get("reject_reason") == "court_depth_reversal"


def test_true_bounce_still_scores_above_threshold():
    """真弹跳:图像 y 局部峰 + court y 继续同向,不许被新守卫误杀。"""
    detector = _detector()
    xs = [300 + 10 * i for i in range(20)]
    ys = [400 + 12 * i if i <= CENTER else 400 + 12 * CENTER - 8 * (i - CENTER)
          for i in range(20)]
    window = _window(xs, ys)
    velocity = _velocity(detector, window)
    # 发球方在远端:球一直朝近端走(court y 单调增),弹跳后略减速
    cy = [10.0 + 0.35 * i if i <= CENTER else 10.0 + 0.35 * CENTER + 0.25 * (i - CENTER)
          for i in range(20)]
    court_window = _window([5.0] * 20, cy)

    score, _ = detector._score_window(window, velocity, CENTER, court_window=court_window)

    assert score >= detector.min_score


def _parabolic_points():
    """帧 1..60,触地在 f30(图像 y 最大):落地前 y=500-2(f-30)^2,反弹后更陡。
    court y 线性 20-0.3f(单调,无反向),x 恒 5。"""
    points = []
    for f in range(1, 61):
        if f <= 30:
            iy = 500.0 - 2.0 * (f - 30) ** 2
        else:
            iy = 500.0 - 5.0 * (f - 30) ** 2
        points.append(TrajectoryPoint(
            frame=f, time_sec=f / 25.0,
            image=[5.0 * f, iy],
            court=[5.0, 20.0 - 0.3 * f],
        ))
    return points


def _contact_points():
    """球拍击球轨迹:图像 y 升到 f30 最大后回升(球落向球拍被打回),court y
    同步在 f30 反转(飞向近端球员再折返)。job 6e98cc64 f28/f142/f217 的形态:
    评分窗口中心偏前几帧时窗口内看不到反转,漏过评分层否决。"""
    points = []
    for f in range(1, 61):
        if f <= 30:
            iy = 400.0 + 8.0 * f
            cy = 10.0 + 0.35 * f
        else:
            iy = 400.0 + 8.0 * 30 - 8.0 * (f - 30)
            cy = 10.0 + 0.35 * 30 - 0.35 * (f - 30)
        points.append(TrajectoryPoint(
            frame=f, time_sec=f / 25.0,
            image=[5.0 * f, iy],
            court=[5.0, cy],
        ))
    return points


def test_refine_events_drops_event_snapped_onto_racket_contact():
    """中心偏前的候选被吸附到击球帧后,最终帧处 court 深度反转 -> 事件整体丢弃,
    不许把击球点画成落点。"""
    detector = _detector()
    points = _contact_points()
    p27 = next(p for p in points if p.frame == 27)
    events = [{
        "frame": 27, "time_sec": p27.time_sec,
        "image": [round(p27.image[0], 2), round(p27.image[1], 2)],
        "court": list(p27.court), "confidence": 0.5,
        "method": "trajectory_lag20", "diagnostics": {},
    }]

    assert detector._refine_events(points, events) == []


def test_finalize_drops_contacts_before_dedupe():
    """次序守护(job 6e98cc64 serve-1 落点两头落空):真弹跳候选(低分)与相邻
    击球点候选(高分)间隔小于 dedupe 最小间隔时,必须先丢击球点再 dedupe,
    否则击球点先吃掉真弹跳、自己再被丢,一个点都不剩。"""
    detector = _detector()
    points = []
    for f in range(1, 61):
        if f <= 25:
            iy = 300.0 + 8.0 * f                # 落向地面,f25 触地(iy 最大 500)
        elif f <= 28:
            iy = 500.0 - 4.0 * (f - 25)         # 反弹上升
        elif f <= 31:
            iy = 488.0 + 5.67 * (f - 28)        # 再落向球拍,f31 接触(iy 505)
        else:
            iy = 505.0 - 7.0 * (f - 31)         # 被击回
        cy = 10.0 + 0.3 * f if f <= 31 else 10.0 + 0.3 * 31 - 0.4 * (f - 31)
        points.append(TrajectoryPoint(frame=f, time_sec=f / 25.0,
                                      image=[5.0 * f, iy], court=[5.0, cy]))

    def _event(frame, confidence):
        p = next(pt for pt in points if pt.frame == frame)
        return {"frame": frame, "time_sec": p.time_sec,
                "image": [round(p.image[0], 2), round(p.image[1], 2)],
                "court": list(p.court), "confidence": confidence,
                "method": "trajectory_lag20", "diagnostics": {}}

    # 击球点分更高;两者间隔 6 帧 < min_event_gap(25fps 下 11 帧)
    refined = detector._refine_events(points, [_event(25, 0.4), _event(31, 0.9)])

    assert [e["frame"] for e in refined] == [25]


def test_refine_events_dedupes_snapped_duplicates():
    """相邻候选吸附到同一触地帧后必须在链内去重——detect 路径不再自己先
    dedupe(先 dedupe 会让击球点吃掉真弹跳,见上一测试),链内不去重就会重复。"""
    detector = _detector()
    points = _parabolic_points()

    def _event(frame, confidence):
        p = next(pt for pt in points if pt.frame == frame)
        return {"frame": frame, "time_sec": p.time_sec,
                "image": [round(p.image[0], 2), round(p.image[1], 2)],
                "court": list(p.court), "confidence": confidence,
                "method": "trajectory_lag20", "diagnostics": {}}

    refined = detector._refine_events(points, [_event(28, 0.5), _event(32, 0.6)])

    assert [e["frame"] for e in refined] == [30]


def test_refine_events_snaps_frame_and_interpolates_court():
    """检测中心偏 2 帧(f28)时:帧吸附到局部图像 y 最大的 f30,court 用亚帧
    精化插值 ~= 真触地时刻的 court(f30 -> cy=11.0),不再是 f28 的单帧投影(11.6)。"""
    detector = _detector()
    points = _parabolic_points()
    p28 = next(p for p in points if p.frame == 28)
    events = [{
        "frame": 28, "time_sec": p28.time_sec,
        "image": [round(p28.image[0], 2), round(p28.image[1], 2)],
        "court": list(p28.court), "confidence": 0.8,
        "method": "trajectory_lag20", "diagnostics": {},
    }]

    refined = detector._refine_events(points, events)

    assert len(refined) == 1
    event = refined[0]
    assert event["frame"] == 30
    assert event["image"][1] == 500.0
    assert abs(event["court"][1] - 11.0) < 0.15
    assert abs(event["court"][0] - 5.0) < 0.01
