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
    """回归测试（controller finding 1b）：ROI 拒绝的检测必须与 visible=False
    一起把 confidence/candidate_count 也清零，不能留下 confidence=0.9、
    candidate_count=1 这种「不可见但仍标着高置信度候选」的不一致状态——
    对齐 TennisBallTracker（tennis_ball.py:70-77）的口径：它的 ROI 过滤在
    `_extract_candidates` 阶段就做了，`candidate_count = len(candidates)`
    天然已经是过 ROI 之后的计数，selected 为 None 时 confidence 恒 None。"""
    detector = TrackNetBallDetector(
        model=_fixed_heatmap_model(peak=(1, 1), value=0.9), input_width=W, input_height=H
    )
    # peak (1,1) 缩放到原分辨率后落在左上角附近；给一个远离该点的 ROI
    far_roi = [(500, 500), (600, 600)]

    point = detector.detect_ball(_blank_frame(), conf=0.5, roi_corners=far_roi)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["image"] is None
    assert detector.last_detection["confidence"] is None
    assert detector.last_detection["candidate_count"] == 0


def test_sub_threshold_heatmap_confidence_is_none():
    """回归测试（controller finding 1a）：热图峰值存在但低于阈值时，
    confidence 必须是 None（不能泄漏原始浮点值），与 TennisBallTracker
    「selected is None ⇒ confidence is None」的口径一致。"""
    detector = TrackNetBallDetector(
        model=_fixed_heatmap_model(peak=(5, 3), value=0.2), input_width=W, input_height=H
    )

    point = detector.detect_ball(_blank_frame(), conf=0.5)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["image"] is None
    assert detector.last_detection["confidence"] is None
    assert detector.last_detection["candidate_count"] == 0


def test_third_frame_uses_real_sliding_window_without_error():
    """3 帧滑窗填满之后（第 3 次真实调用）仍应正常工作，且不再需要复制填充。"""
    detector = TrackNetBallDetector(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    frame = _blank_frame()
    for _ in range(3):
        point = detector.detect_ball(frame, conf=0.5)
    assert point != [0, 0]


def test_stacked_model_input_shape_dtype_and_channel_order():
    """回归测试（controller finding 2）：验证喂给模型的堆叠输入张量形状/
    dtype/通道堆叠顺序符合 _build_model_input 的实现约定——3 帧沿 channel
    维堆叠成 [9,H,W]、float 类型、newest-first（channel 0-2 是最新帧，
    3-5 中间帧，6-8 最早帧；见该方法内的注释：对齐上游 infer_on_video.py
    的 `concatenate((img, img_prev, img_preprev))` 顺序）。"""
    captured = {}

    def capturing_model(stacked_input):
        captured["stacked_input"] = stacked_input
        return np.zeros((H, W), dtype=np.float32)

    detector = TrackNetBallDetector(model=capturing_model, input_width=W, input_height=H)

    # 3 帧视觉上可分辨的常量值帧（已经是 input_width x input_height，
    # cv2.resize 是恒等变换，避免插值噪声干扰通道值断言）
    oldest = np.full((H, W, 3), int(0.1 * 255), dtype=np.uint8)
    middle = np.full((H, W, 3), int(0.5 * 255), dtype=np.uint8)
    newest = np.full((H, W, 3), int(0.9 * 255), dtype=np.uint8)

    detector.detect_ball(oldest, conf=0.5)
    detector.detect_ball(middle, conf=0.5)
    detector.detect_ball(newest, conf=0.5)  # 此时窗口已满且全是真实帧，无复制填充

    stacked = captured["stacked_input"]
    assert stacked.shape == (9, H, W)
    assert np.issubdtype(stacked.dtype, np.floating)

    assert stacked[0:3].mean() == pytest.approx(0.9, abs=0.01)  # newest
    assert stacked[3:6].mean() == pytest.approx(0.5, abs=0.01)  # middle
    assert stacked[6:9].mean() == pytest.approx(0.1, abs=0.01)  # oldest
