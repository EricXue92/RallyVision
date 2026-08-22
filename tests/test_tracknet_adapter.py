import numpy as np
import pytest

from tennis_analysis.detection.tracknet_ball import TrackNetBallDetector

# 微缩输入分辨率：测试只验证滑窗/接口契约，无需真实 640x360 权重网格
W, H = 16, 12


def _fixed_heatmap_model(peak=(5, 3), value=0.9):
    """返回固定热图的假模型：忽略输入，在 (x=peak[0], y=peak[1]) 处放一个高于
    阈值的峰值，其余为 0。用于验证滑窗/接口契约而不依赖 torch/权重。"""
    heatmap = np.zeros((H, W), dtype=np.float32)
    heatmap[peak[1], peak[0]] = value

    def model_fn(_stacked_input):
        return heatmap

    return model_fn


def _zero_heatmap_model(_stacked_input):
    return np.zeros((H, W), dtype=np.float32)


def _blank_frame():
    return np.zeros((H * 10, W * 10, 3), dtype=np.uint8)


def test_first_two_frames_still_produce_result():
    """不足 3 帧时应复制首帧填充窗口内部；前两帧调用也要能出结果，不能因窗口
    未满而报错或恒返回 [0, 0]。"""
    detector = TrackNetBallDetector(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    frame = _blank_frame()

    for i in range(2):
        point = detector.detect_ball(frame, conf=0.5)
        assert point != [0, 0], f"frame {i} 未产生检测结果"
        assert detector.last_detection["visible"] is True


def test_last_detection_has_complete_keys():
    detector = TrackNetBallDetector(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    detector.detect_ball(_blank_frame(), conf=0.5)
    expected_keys = {"visible", "accepted", "image", "confidence", "candidate_count"}
    assert set(detector.last_detection.keys()) == expected_keys


def test_all_zero_heatmap_returns_origin_and_invisible():
    detector = TrackNetBallDetector(model=_zero_heatmap_model, input_width=W, input_height=H)

    point = detector.detect_ball(_blank_frame(), conf=0.5)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["image"] is None
    assert detector.last_detection["candidate_count"] == 0


def test_requires_model_or_model_path():
    with pytest.raises(ValueError):
        TrackNetBallDetector()


def test_roi_rejects_point_outside_expanded_box():
    detector = TrackNetBallDetector(
        model=_fixed_heatmap_model(peak=(1, 1), value=0.9), input_width=W, input_height=H
    )
    # peak (1,1) 缩放到原分辨率后落在左上角附近；给一个远离该点的 ROI
    far_roi = [(500, 500), (600, 600)]

    point = detector.detect_ball(_blank_frame(), conf=0.5, roi_corners=far_roi)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False


def test_third_frame_uses_real_sliding_window_without_error():
    """3 帧滑窗填满之后（第 3 次真实调用）仍应正常工作，且不再需要复制填充。"""
    detector = TrackNetBallDetector(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    frame = _blank_frame()
    for _ in range(3):
        point = detector.detect_ball(frame, conf=0.5)
    assert point != [0, 0]
