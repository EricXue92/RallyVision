"""远场 ROI 二次推理（CourtCheck infer_with_far_roi 思想的移植）单测。

整帧压到 640x360 后远半场的球只剩 1~2 像素，是发球/深球落点缺失的主因。
detect_ball 增加 far_roi_rect 参数：除全帧推理外，把远半场区域裁剪出来单独
resize 再推理一次，融合规则——全帧漏检时用远场结果补洞；两边都检到且全帧
结果落在远场区域内时优先远场结果（有效分辨率更高、坐标更准）。

测试沿用 test_tracknet_adapter.py 的假模型注入手法（微缩分辨率,无需权重）；
detect_ball 传 far_roi_rect 时假模型每帧被调用两次,约定顺序:先全帧后远场。
"""
import numpy as np

from tennis_analysis.detection.tracknet_ball import (
    TrackNetBallDetector,
    TrackNetBallTrackerAdapter,
    compute_far_roi_rect,
)

# 微缩输入分辨率（同 test_tracknet_adapter.py）
W, H = 16, 12
FRAME_W, FRAME_H = 160, 120  # 全帧峰值 -> 帧坐标的缩放是 10x


def _frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _heatmap(peak, value=0.9):
    heatmap = np.zeros((H, W), dtype=np.float32)
    if peak is not None:
        heatmap[peak[1], peak[0]] = value
    return heatmap


class _SequenceModel:
    """按调用次序返回给定热图序列的假模型，并记录调用次数。
    传 far_roi_rect 时约定每帧两次调用：第 1 次全帧、第 2 次远场裁剪。"""

    def __init__(self, heatmaps):
        self._heatmaps = list(heatmaps)
        self.calls = 0

    def __call__(self, _stacked_input):
        heatmap = self._heatmaps[min(self.calls, len(self._heatmaps) - 1)]
        self.calls += 1
        return heatmap


# ---------------------------------------------------------------- 纯函数：远场矩形

def test_compute_far_roi_rect_pads_bbox():
    points = [(300, 100), (980, 100), (200, 300), (1080, 300)]
    # bbox x:200..1080 (w=880) pad 10% = 88；y:100..300 (h=200) 上 35%=70 下 10%=20
    assert compute_far_roi_rect(points, 1280, 720) == (112, 30, 1168, 320)


def test_compute_far_roi_rect_clamps_to_frame():
    points = [(20, 10), (1260, 10), (10, 300), (1270, 300)]
    x1, y1, x2, y2 = compute_far_roi_rect(points, 1280, 720)
    assert x1 == 0 and y1 == 0 and x2 == 1280
    assert 0 < y2 <= 720


def test_compute_far_roi_rect_degenerate_returns_none():
    assert compute_far_roi_rect([], 1280, 720) is None
    assert compute_far_roi_rect([(100, 100), (200, 100)], 1280, 720) is None  # 点数不足
    nan_points = [(np.nan, np.nan), (100, 100), (200, 100), (300, 100)]
    assert compute_far_roi_rect(nan_points, 1280, 720) is None  # 含 NaN 剔除后不足
    flat = [(100, 100), (200, 100), (300, 100), (400, 100)]
    assert compute_far_roi_rect(flat, 1280, 720) is None  # 零高度


# ---------------------------------------------------------------- 融合规则

def test_far_roi_fills_full_frame_miss():
    """全帧漏检、远场检到 → 用远场结果补洞，坐标按裁剪矩形回映射。"""
    far_rect = (80, 20, 160, 80)  # w=80 -> 每模型像素 5px；h=60 -> 每像素 5px
    model = _SequenceModel([_heatmap(None), _heatmap((5, 3))])
    detector = TrackNetBallDetector(model=model, input_width=W, input_height=H)

    point = detector.detect_ball(_frame(), conf=0.5, far_roi_rect=far_rect)

    assert model.calls == 2
    assert point == [80 + 5 * 5, 20 + 3 * 5]  # [105, 35]
    assert detector.last_detection["visible"] is True


def test_far_result_preferred_when_full_point_inside_far_region():
    """两边都检到且全帧结果落在远场矩形内 → 远场坐标更准，优先远场。"""
    far_rect = (40, 24, 120, 72)  # 全帧峰 (5,3)->帧坐标 (50,30)，在矩形内
    model = _SequenceModel([_heatmap((5, 3)), _heatmap((8, 6))])
    detector = TrackNetBallDetector(model=model, input_width=W, input_height=H)

    point = detector.detect_ball(_frame(), conf=0.5, far_roi_rect=far_rect)

    assert point == [40 + 8 * 5, 24 + 6 * 4]  # [80, 48]


def test_full_result_wins_when_outside_far_region():
    """全帧结果在近场（远场矩形外）→ 远场裁剪里的峰值是杂讯，用全帧结果。"""
    far_rect = (100, 10, 148, 58)  # 全帧峰 (5,3)->帧坐标 (50,30)，在矩形外
    model = _SequenceModel([_heatmap((5, 3)), _heatmap((8, 6))])
    detector = TrackNetBallDetector(model=model, input_width=W, input_height=H)

    point = detector.detect_ball(_frame(), conf=0.5, far_roi_rect=far_rect)

    assert point == [50, 30]


def test_no_far_rect_keeps_single_inference():
    model = _SequenceModel([_heatmap((5, 3))])
    detector = TrackNetBallDetector(model=model, input_width=W, input_height=H)

    point = detector.detect_ball(_frame(), conf=0.5)

    assert model.calls == 1
    assert point == [50, 30]


def test_system_far_roi_rect_derived_from_court_mapper(monkeypatch):
    """system._far_roi_rect_for_ball：远场矩形 = 远端两底线角（image_court_corners
    前两点）+ 球网两端图像坐标 交给 compute_far_roi_rect。bare __new__ 手法。"""
    import tennis_analysis.system as system_module
    from tennis_analysis.system import TennisAnalysisSystem
    from tennis_analysis.court.mapper import CourtMapper

    monkeypatch.setattr(system_module, "compute_far_roi_rect", compute_far_roi_rect, raising=False)

    corners = [(300, 100), (980, 100), (1180, 620), (100, 620)]  # TL TR BR BL
    mapper = CourtMapper(corners)
    net_y = mapper.court_dimensions[1] / 2
    net_left = mapper.court_to_image((0.0, net_y))
    net_right = mapper.court_to_image((mapper.court_dimensions[0], net_y))
    expected = compute_far_roi_rect(
        [corners[0], corners[1], tuple(net_left), tuple(net_right)], 1280, 720
    )
    assert expected is not None  # 测试前提：合法梯形必须能算出矩形

    system = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    system.far_roi = True
    system.ball_detector = "tracknet"
    system.court_mapper = mapper
    system.frame_width, system.frame_height = 1280, 720

    assert system._far_roi_rect_for_ball() == expected


def test_system_far_roi_rect_disabled_or_unavailable(monkeypatch):
    import tennis_analysis.system as system_module
    from tennis_analysis.system import TennisAnalysisSystem
    from tennis_analysis.court.mapper import CourtMapper

    monkeypatch.setattr(system_module, "compute_far_roi_rect", compute_far_roi_rect, raising=False)

    system = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    system.far_roi = False
    system.ball_detector = "tracknet"
    system.court_mapper = CourtMapper([(300, 100), (980, 100), (1180, 620), (100, 620)])
    system.frame_width, system.frame_height = 1280, 720
    assert system._far_roi_rect_for_ball() is None  # 开关关闭

    system2 = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    system2.far_roi = True
    system2.ball_detector = "tracknet"
    system2.court_mapper = None
    system2.frame_width, system2.frame_height = 1280, 720
    assert system2._far_roi_rect_for_ball() is None  # 尚无球场标定


def test_adapter_forwards_far_roi_rect():
    far_rect = (80, 20, 160, 80)
    model = _SequenceModel([_heatmap(None), _heatmap((5, 3))])
    adapter = TrackNetBallTrackerAdapter(
        TrackNetBallDetector(model=model, input_width=W, input_height=H)
    )

    point = adapter.detect_ball(_frame(), far_roi_rect=far_rect)

    assert model.calls == 2
    assert point == [105, 35]
