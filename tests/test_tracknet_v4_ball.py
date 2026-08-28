"""TrackNetV4 后端契约测试。

覆盖三件容易悄悄写错、写错了只表现为「检测变差」而不报错的事：
1. 9 通道重排（BGR 新->旧  →  RGB 旧->新）——顺序错了运动注意力图就是反的；
2. 峰值加权质心精化——sigmoid 热图会饱和成一片，纯 argmax 会在片内抖动；
3. BatchNorm 归一化维是 W 不是 C（上游 Keras 的历史怪癖，见模块 docstring）。

不依赖真实权重：结构类按需构造，检测逻辑用注入的 fake model。
"""
import numpy as np
import pytest

from tennis_analysis.detection.tracknet_ball import TrackNetBallDetector
from tennis_analysis.detection.tracknet_v4_ball import (
    _BGR_NEWEST_FIRST_TO_RGB_OLDEST_FIRST,
    TrackNetV4BallDetector,
    get_tracknet_v4_class,
)

# 微缩分辨率：契约测试不需要真实网格
W, H = 16, 12


def _frame(fill):
    """整帧填同一个 BGR 三元组，便于按通道值反查它来自哪一帧。"""
    frame = np.zeros((H * 4, W * 4, 3), dtype=np.uint8)
    frame[:, :] = fill
    return frame


def test_channel_reorder_yields_rgb_oldest_first():
    """基类堆出来的是「BGR + 时间逆序」，上游 TrackNetV4 要的是「RGB + 时间正序」。

    三帧各给一个可区分的 BGR 值，重排后逐通道核对：前 3 通道必须是最旧那帧的
    R、G、B。这条断言挂了说明重排常量被改坏，模型会收到时间反着的输入。
    """
    detector = TrackNetBallDetector(model=lambda _: np.zeros((H, W), np.float32),
                                    input_width=W, input_height=H)
    oldest, middle, newest = (10, 20, 30), (40, 50, 60), (70, 80, 90)  # 各帧的 (B, G, R)
    for fill in (oldest, middle, newest):
        detector._update_frame_window(_frame(fill))

    stacked = detector._build_model_input()
    reordered = stacked[_BGR_NEWEST_FIRST_TO_RGB_OLDEST_FIRST]
    got = [float(reordered[channel, 0, 0] * 255.0) for channel in range(9)]
    expected = [
        oldest[2], oldest[1], oldest[0],
        middle[2], middle[1], middle[0],
        newest[2], newest[1], newest[0],
    ]
    assert got == pytest.approx(expected, abs=1e-3)


def test_saturated_plateau_resolves_to_plateau_center():
    """球心附近 sigmoid 会饱和成一小片 1.0：取加权质心而不是 argmax，
    否则峰值会在片内任意跳，映回原图就是几像素的落点抖动。"""
    heatmap = np.zeros((H, W), dtype=np.float32)
    heatmap[4:7, 6:9] = 1.0  # 3x3 饱和片，中心 (x=7, y=5)
    detector = TrackNetV4BallDetector(model=lambda _: heatmap, input_width=W, input_height=H)

    frame = _frame((0, 0, 0))
    detector.detect_ball(frame)
    scale_x, scale_y = frame.shape[1] / W, frame.shape[0] / H
    assert detector.get_last_detection()["image"] == [int(7 * scale_x), int(5 * scale_y)]


def test_peak_below_threshold_is_not_reported():
    heatmap = np.full((H, W), 0.2, dtype=np.float32)
    detector = TrackNetV4BallDetector(model=lambda _: heatmap, input_width=W, input_height=H)

    assert detector.detect_ball(_frame((0, 0, 0)), conf=0.5) == [0, 0]
    assert detector.get_last_detection()["visible"] is False


def test_batchnorm_params_are_sized_by_width_not_channels():
    """归一化维是 W——参数长度必须等于该层的宽。写成 nn.BatchNorm2d（按通道）
    形状能过一半、语义全错，所以这里逐层钉死。"""
    # 宽取 96：三次下采样得 96/48/24/12，与任何一层的通道数（64/128/256/512）
    # 都不相等——宽和通道数撞上的话，这条测试对「误用 BatchNorm2d」就没有区分力了
    net = get_tracknet_v4_class()(fusion="A", input_height=32, input_width=96)
    for index, expected_width in ((0, 96), (2, 48), (4, 24), (7, 12), (10, 24), (13, 48), (16, 96)):
        gamma = getattr(net, f"bn{index}").gamma
        assert gamma.shape == (expected_width,), f"bn{index} 归一化维错了：{gamma.shape}"


@pytest.mark.parametrize("fusion", ["A", "B", "none"])
def test_forward_shape_and_range(fusion):
    """三种融合都要吐 (B, 3, H, W) 的 0-1 热图（每个输入帧一张）。"""
    import torch

    net = get_tracknet_v4_class()(fusion=fusion, input_height=32, input_width=64).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 9, 32, 64))
    assert out.shape == (1, 3, 32, 64)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_missing_model_and_path_raises():
    with pytest.raises(ValueError):
        TrackNetV4BallDetector()
