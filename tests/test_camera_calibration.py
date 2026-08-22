"""Task 10 amended brief：固定机位标定的纯逻辑单测（不跑视频/不碰 cv2/torch）。

覆盖 tennis_analysis.court.camera_calibration 两个纯函数：
- median_keypoints_over_frames：逐点中位数聚合 + <3 帧有效即判整体无效
- keypoints_drifted：漂移守卫判定（>=4 个可比较关键点偏移 >10px 才判漂移）
"""
import numpy as np

from tennis_analysis.court.camera_calibration import keypoints_drifted, median_keypoints_over_frames

BASE = np.array([[float(i) * 10.0, float(i) * 20.0] for i in range(14)])


def _frame(nan_indices=()):
    pts = BASE.copy()
    for idx in nan_indices:
        pts[idx] = [np.nan, np.nan]
    return pts


def test_median_ignores_keypoint_valid_in_fewer_than_min_frames():
    # 关键点 0 只在 2/4 帧里有效（<3）-> 整体判无效；其余关键点全部 4 帧都有效。
    frames = [_frame(), _frame(nan_indices=[0]), _frame(nan_indices=[0]), _frame()]
    median, mask = median_keypoints_over_frames(frames, min_valid_frames=3)
    assert not mask[0]
    assert np.isnan(median[0]).all()
    assert mask[1]
    assert np.allclose(median[1], BASE[1])


def test_median_is_elementwise_median_with_noise():
    rng = np.random.default_rng(0)
    frames = [BASE + rng.normal(0, 0.5, BASE.shape) for _ in range(5)]
    median, mask = median_keypoints_over_frames(frames, min_valid_frames=3)
    assert mask.all()
    assert np.allclose(median, BASE, atol=2.0)


def _baseline():
    return median_keypoints_over_frames([BASE, BASE, BASE], min_valid_frames=3)


def test_no_drift_when_all_deltas_within_threshold():
    baseline_median, baseline_mask = _baseline()
    current = BASE + 2.0  # 全部只偏 2px（<10px 阈值）
    assert keypoints_drifted(current, baseline_median, baseline_mask) is False


def test_drift_detected_when_four_keypoints_shift_beyond_threshold():
    baseline_median, baseline_mask = _baseline()
    current = BASE.copy()
    for idx in range(4):
        current[idx] = current[idx] + [20.0, 0.0]
    assert keypoints_drifted(current, baseline_median, baseline_mask) is True


def test_no_drift_when_only_three_keypoints_shift_beyond_threshold():
    baseline_median, baseline_mask = _baseline()
    current = BASE.copy()
    for idx in range(3):  # 差 1 个才达到 min_keypoints=4 的阈值
        current[idx] = current[idx] + [20.0, 0.0]
    assert keypoints_drifted(current, baseline_median, baseline_mask) is False


def test_nan_keypoints_in_current_frame_excluded_from_drift_count():
    baseline_median, baseline_mask = _baseline()
    current = BASE.copy()
    for idx in range(4):
        current[idx] = current[idx] + [20.0, 0.0]
    current[0] = [np.nan, np.nan]  # 本会漂移的一个点这一帧没检测到 -> 不计入比较
    # 剩 3 个真正超阈值的点 -> 不足 min_keypoints=4 -> 不判漂移
    assert keypoints_drifted(current, baseline_median, baseline_mask) is False


def test_drift_ignores_keypoints_invalid_in_baseline():
    # 关键点 0 在基线里本就无效（mask=False）；即使当前帧该点偏移巨大也不计入比较。
    baseline_median, baseline_mask = _baseline()
    baseline_mask = baseline_mask.copy()
    baseline_mask[0:4] = False  # 只留 10 个可比较点，且都不偏移
    current = BASE.copy()
    current[0] = current[0] + [1000.0, 1000.0]
    assert keypoints_drifted(current, baseline_median, baseline_mask) is False
