import numpy as np
import pytest

from tennis_analysis.detection.tracknet_ball import (
    TrackNetBallDetector,
    is_implausible_ball_jump,
)
from tennis_analysis.detection.wasb_ball import WASBBallDetector

# 微缩输入分辨率：测试只验证滑窗/接口契约，无需真实权重网格
W, H = 16, 12

# Task 9b：契约测试参数化覆盖两个后端类（TrackNet 与 WASB 接口完全对齐）
DETECTOR_CLASSES = [TrackNetBallDetector, WASBBallDetector]
DETECTOR_IDS = ["tracknet", "wasb"]


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


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_first_two_frames_still_produce_result(detector_cls):
    """不足 3 帧时应复制首帧填充窗口内部；前两帧调用也要能出结果，不能因窗口
    未满而报错或恒返回 [0, 0]。"""
    detector = detector_cls(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    frame = _blank_frame()

    for i in range(2):
        point = detector.detect_ball(frame, conf=0.5)
        assert point != [0, 0], f"frame {i} 未产生检测结果"
        assert detector.last_detection["visible"] is True


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_last_detection_has_complete_keys(detector_cls):
    detector = detector_cls(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    detector.detect_ball(_blank_frame(), conf=0.5)
    expected_keys = {"visible", "accepted", "image", "confidence", "candidate_count"}
    assert set(detector.last_detection.keys()) == expected_keys


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_all_zero_heatmap_returns_origin_and_invisible(detector_cls):
    detector = detector_cls(model=_zero_heatmap_model, input_width=W, input_height=H)

    point = detector.detect_ball(_blank_frame(), conf=0.5)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["image"] is None
    assert detector.last_detection["candidate_count"] == 0


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_requires_model_or_model_path(detector_cls):
    with pytest.raises(ValueError):
        detector_cls()


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_roi_rejects_point_outside_expanded_box(detector_cls):
    """回归测试（controller finding 1b）：ROI 拒绝的检测必须与 visible=False
    一起把 confidence/candidate_count 也清零，不能留下 confidence=0.9、
    candidate_count=1 这种「不可见但仍标着高置信度候选」的不一致状态——
    对齐 TennisBallTracker（tennis_ball.py:70-77）的口径：它的 ROI 过滤在
    `_extract_candidates` 阶段就做了，`candidate_count = len(candidates)`
    天然已经是过 ROI 之后的计数，selected 为 None 时 confidence 恒 None。"""
    detector = detector_cls(model=_fixed_heatmap_model(peak=(1, 1), value=0.9), input_width=W, input_height=H)
    # peak (1,1) 缩放到原分辨率后落在左上角附近；给一个远离该点的 ROI
    far_roi = [(500, 500), (600, 600)]

    point = detector.detect_ball(_blank_frame(), conf=0.5, roi_corners=far_roi)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["image"] is None
    assert detector.last_detection["confidence"] is None
    assert detector.last_detection["candidate_count"] == 0


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_sub_threshold_heatmap_confidence_is_none(detector_cls):
    """回归测试（controller finding 1a）：热图峰值存在但低于阈值时，
    confidence 必须是 None（不能泄漏原始浮点值），与 TennisBallTracker
    「selected is None ⇒ confidence is None」的口径一致。"""
    detector = detector_cls(model=_fixed_heatmap_model(peak=(5, 3), value=0.2), input_width=W, input_height=H)

    point = detector.detect_ball(_blank_frame(), conf=0.5)

    assert point == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["image"] is None
    assert detector.last_detection["confidence"] is None
    assert detector.last_detection["candidate_count"] == 0


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_third_frame_uses_real_sliding_window_without_error(detector_cls):
    """3 帧滑窗填满之后（第 3 次真实调用）仍应正常工作，且不再需要复制填充。"""
    detector = detector_cls(model=_fixed_heatmap_model(), input_width=W, input_height=H)
    frame = _blank_frame()
    for _ in range(3):
        point = detector.detect_ball(frame, conf=0.5)
    assert point != [0, 0]


def test_stacked_model_input_shape_dtype_and_channel_order_tracknet():
    """回归测试（controller finding 2）：验证喂给模型的堆叠输入张量形状/
    dtype/通道堆叠顺序符合 TrackNetBallDetector._build_model_input 的实现
    约定——3 帧沿 channel 维堆叠成 [9,H,W]、float 类型、newest-first（channel
    0-2 是最新帧，3-5 中间帧，6-8 最早帧；见该方法内的注释：对齐上游
    infer_on_video.py 的 `concatenate((img, img_prev, img_preprev))` 顺序）。
    """
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


def test_stacked_model_input_shape_dtype_and_channel_order_wasb():
    """WASB 的堆叠顺序与 TrackNet 相反：oldest-first（channel 0-2 最早帧，
    6-8 最新帧），核对自上游 src/datasets/tennis.py + dataloaders/
    dataset_loader.py 的窗口构造顺序（见 wasb_ball.py 模块 docstring"输入
    约定"节）。WASB 还会做 BGR→RGB + ImageNet 归一化（不是简单 /255），所以
    这里只断言形状/dtype/相对大小关系（值经归一化后不再是 0.1/0.5/0.9），
    不复用 TrackNet 那组精确均值断言。
    """
    captured = {}

    def capturing_model(stacked_input):
        captured["stacked_input"] = stacked_input
        return np.zeros((H, W), dtype=np.float32)

    detector = WASBBallDetector(model=capturing_model, input_width=W, input_height=H)

    oldest = np.full((H, W, 3), int(0.1 * 255), dtype=np.uint8)
    middle = np.full((H, W, 3), int(0.5 * 255), dtype=np.uint8)
    newest = np.full((H, W, 3), int(0.9 * 255), dtype=np.uint8)

    detector.detect_ball(oldest, conf=0.5)
    detector.detect_ball(middle, conf=0.5)
    detector.detect_ball(newest, conf=0.5)  # 此时窗口已满且全是真实帧，无复制填充

    stacked = captured["stacked_input"]
    assert stacked.shape == (9, H, W)
    assert np.issubdtype(stacked.dtype, np.floating)

    # 归一化后仍保持单调：oldest(最暗) < middle < newest(最亮) 的均值序关系
    assert stacked[0:3].mean() < stacked[3:6].mean() < stacked[6:9].mean()


# ---------------------------------------------------------------------------
# Task 9b Step 1: 球员框 gating 纯函数测试（tracknet_ball.py::is_implausible_ball_jump）
# ---------------------------------------------------------------------------


def test_gating_rejects_jump_far_from_players():
    """突变（远超 150px 阈值）且距两名球员中心都超过 300px → 判误检丢弃。"""
    last_point = (100, 100)
    candidate = (600, 100)  # 距 last_point 500px，远超 150px 阈值
    player_centers = [(50, 50), (150, 150)]  # 均远离 candidate（>300px）

    assert is_implausible_ball_jump(candidate, last_point, player_centers, 1280, 720) is True


def test_gating_accepts_jump_near_player():
    """突变但落在某球员中心附近（合法击球点）→ 放行。"""
    last_point = (100, 100)
    candidate = (600, 100)  # 距 last_point 500px，是跳变
    player_centers = [(620, 110), (50, 50)]  # 第一个球员离 candidate 很近

    assert is_implausible_ball_jump(candidate, last_point, player_centers, 1280, 720) is False


def test_gating_accepts_without_last_point():
    """无上一帧接受点（首帧/回合刚开始）→ 不构成跳变判定基础，直接放行。"""
    candidate = (600, 100)
    player_centers = [(50, 50), (150, 150)]  # 均远离 candidate，若判定跳变本会被拒

    assert is_implausible_ball_jump(candidate, None, player_centers, 1280, 720) is False


def test_gating_accepts_without_player_centers():
    """拿不到球员数据（None 或空列表）时 gating 直接放行，不因缺球员检测数据丢球。"""
    last_point = (100, 100)
    candidate = (600, 100)  # 是跳变，但没有球员数据可比对

    assert is_implausible_ball_jump(candidate, last_point, None, 1280, 720) is False
    assert is_implausible_ball_jump(candidate, last_point, [], 1280, 720) is False


def test_gating_ignores_none_entries_in_player_centers():
    """player_centers 里某侧球员本帧未检出（None）应被跳过，不参与距离比较、
    不导致异常。"""
    last_point = (100, 100)
    candidate = (600, 100)
    player_centers = [None, (50, 50)]  # 唯一非 None 的球员也远离 candidate

    assert is_implausible_ball_jump(candidate, last_point, player_centers, 1280, 720) is True


def test_gating_small_movement_is_never_a_jump():
    """距离未超过跳变阈值本就不算跳变，即便远离所有球员也应放行（正常连续轨迹）。"""
    last_point = (100, 100)
    candidate = (110, 105)  # 距 last_point ~11px，远小于 150px 阈值
    player_centers = [(900, 900)]  # 远离，但因为不是跳变所以不影响判定

    assert is_implausible_ball_jump(candidate, last_point, player_centers, 1280, 720) is False


def test_gating_threshold_scales_with_resolution():
    """阈值按分辨率相对 1280x720 缩放：同样的像素位移在低分辨率下应更容易
    被判定为「跳变」（缩放后阈值更小）。"""
    last_point = (50, 50)
    candidate = (130, 50)  # 距 last_point 80px
    player_centers = [(900, 900)]  # 远离

    # 在 1280x720 基准下 80px < 150px 阈值，不算跳变，放行
    assert is_implausible_ball_jump(candidate, last_point, player_centers, 1280, 720) is False
    # 在 640x360（缩放系数 0.5）下，跳变阈值缩到 75px，80px 距离超过阈值，
    # 且 player_centers 唯一球员依然远离（缩放后 proximity 阈值 150px，
    # 900,900 距 candidate 仍远超）→ 判误检丢弃
    assert is_implausible_ball_jump(candidate, last_point, player_centers, 640, 360) is True


# ---------------------------------------------------------------------------
# Task 9b Step 1: gating 接入两后端 detect_ball 路径的集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_detect_ball_gating_rejects_jump_far_from_players(detector_cls):
    """建立一个「已接受」的基准点后，下一帧候选点跳变且远离所有 player_centers
    → detect_ball 应把该帧视为不可见（gating 在 detect_ball 内部生效）。"""
    # 微缩分辨率 WxH=16x12，原始帧放大到 160x120（10x），以下坐标都在原始帧空间
    detector = detector_cls(model=_fixed_heatmap_model(peak=(1, 1), value=0.9), input_width=W, input_height=H)
    frame = _blank_frame()  # 160x120

    # 第一次检测：peak (1,1) 缩放到原始帧空间约 (10,10)，建立 last_accepted_point
    first = detector.detect_ball(frame, conf=0.5)
    assert first != [0, 0]

    # 第二次检测：换一个远离 (10,10) 的峰值，且不提供任何靠近它的 player_centers
    # （160x120 帧相对 1280x720 基准缩放系数很小，300px 阈值缩到 ~44px；下面两个
    # 球员中心离新候选点 (140,100) 都超过 44px，应判定为误检丢弃）
    detector._model_fn = _fixed_heatmap_model(peak=(14, 10), value=0.9)  # 缩放后约 (140,100)
    far_players = [(5, 5), (12, 8)]

    second = detector.detect_ball(frame, conf=0.5, player_centers=far_players)

    assert second == [0, 0]
    assert detector.last_detection["visible"] is False
    assert detector.last_detection["confidence"] is None
    assert detector.last_detection["candidate_count"] == 0


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_detect_ball_gating_accepts_jump_near_player(detector_cls):
    """同样的跳变，如果 player_centers 里有一个点靠近新候选点，应放行
    （合法击球场景）。"""
    detector = detector_cls(model=_fixed_heatmap_model(peak=(1, 1), value=0.9), input_width=W, input_height=H)
    frame = _blank_frame()

    first = detector.detect_ball(frame, conf=0.5)
    assert first != [0, 0]

    detector._model_fn = _fixed_heatmap_model(peak=(14, 10), value=0.9)  # 缩放后约 (140,100)
    near_players = [(145, 105)]  # 紧贴新候选点

    second = detector.detect_ball(frame, conf=0.5, player_centers=near_players)

    assert second != [0, 0]
    assert detector.last_detection["visible"] is True


@pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES, ids=DETECTOR_IDS)
def test_detect_ball_gating_passthrough_without_player_centers(detector_cls):
    """不传 player_centers（现有调用点默认行为）时 gating 不生效，跳变候选照常
    通过——回归防护：不能因为新增 gating 就默默丢掉旧调用点的检测结果。"""
    detector = detector_cls(model=_fixed_heatmap_model(peak=(1, 1), value=0.9), input_width=W, input_height=H)
    frame = _blank_frame()

    first = detector.detect_ball(frame, conf=0.5)
    assert first != [0, 0]

    detector._model_fn = _fixed_heatmap_model(peak=(14, 10), value=0.9)
    second = detector.detect_ball(frame, conf=0.5)  # 不传 player_centers

    assert second != [0, 0]
    assert detector.last_detection["visible"] is True
