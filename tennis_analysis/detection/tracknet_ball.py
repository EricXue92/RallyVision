"""TrackNet 网球检测封装（--ball-detector tracknet 后端）。

模型：yastrebksv/TrackNet（Apache-2.0）
https://github.com/yastrebksv/TrackNet
权重下载见 weights/README.md；网络结构 (BallTrackerNet) 抄自该仓库的
model.py（Apache-2.0，仅保留结构定义，不含训练代码）。

与球场关键点检测器共享同名结构（court/keypoint_detector.py 的
BallTrackerNet），但两者是不同上游仓库的不同网络实例，**不能直接复用**
court 那份 `_get_ball_tracker_net_class()`：
  - 球检测 in_channels=9（3 帧 RGB 沿 channel 堆叠）、
    out_channels=256（逐像素做 256 类分类以重建灰度热图，TrackNetV2 的
    训练技巧——每个像素的输出是"该像素强度属于哪一类(0-255)"的分类，
    不是该点是否是球心的直接概率）。
  - 球场检测 in_channels=3（单帧）、out_channels=15（15 个关键点各一个
    独立热图通道）。
  conv1 的 in_channels 不同导致权重形状不兼容，court 那份函数把
  in_channels 硬编码成 3，故这里单独复制一份网络结构（Task 9 binding
  说明："if they differ, copy separately with attribution"）。

后处理与上游 infer_on_video.py 的 postprocess()（阈值二值化 +
cv2.HoughCircles 检圆）不同：本项目采用与 CourtKeypointDetector 一致的
更简单方案——对逐像素 argmax 重建出的灰度热图取全局峰值 + 阈值判断。
上游 forward() 末尾会把输出 reshape 成 (batch, out_channels, -1) 并在
testing=True 时接一次 softmax；这两步都不含可训练参数，纯粹是给分类
loss/后处理用的形状变换，这里省略，直接在卷积输出的原始空间张量
(batch, 256, H, W) 上逐像素 argmax，数值等价。
"""
import time
from collections import deque

import cv2
import numpy as np

# 上游训练/推理输入分辨率
_MODEL_INPUT_WIDTH = 640
_MODEL_INPUT_HEIGHT = 360
_WINDOW_SIZE = 3
_DEFAULT_CONF_THRESHOLD = 0.5


class TrackNetBallDetector:
    """TrackNet 球检测封装，接口对齐 detection/tennis_ball.py::TennisBallTracker。

    内部维护一个长度为 3 的滑窗（最近 3 帧，resize 到 640x360 后沿 channel
    堆叠为 [9,H,W] 送入网络）；不足 3 帧时（视频刚开始）复制已知最早的一帧
    填满窗口。

    支持注入假模型用于测试，无需 torch/权重：
        TrackNetBallDetector(model_path=None, model=fake_callable)
    其中 fake_callable(stacked_input: np.ndarray[9,H,W]) -> np.ndarray[H,W]
    （灰度热图，值域任意，内部会做全局峰值 + 阈值判断；真实模型适配器把
    torch 推理结果归一化到 0-1 范围后再喂给下游同一套峰值判断逻辑）。
    """

    def __init__(
        self,
        model_path=None,
        model=None,
        device=None,
        input_width=_MODEL_INPUT_WIDTH,
        input_height=_MODEL_INPUT_HEIGHT,
        roi_padding_ratio=0.08,
        show_performance_stats=False,
    ):
        if model is None and model_path is None:
            raise ValueError(
                "TrackNetBallDetector 需要 model_path（加载真实权重）或 model"
                "（注入可调用对象，通常用于测试）之一 / "
                "TrackNetBallDetector requires either model_path or an injected model callable"
            )

        self.input_width = input_width
        self.input_height = input_height
        self.roi_padding_ratio = roi_padding_ratio
        self.show_performance_stats = show_performance_stats

        self._frame_window = deque(maxlen=_WINDOW_SIZE)
        self.last_detection = self._empty_detection_state()

        if model is not None:
            self._model_fn = model
        else:
            self._model_fn = _TorchBallTrackerNetAdapter(model_path, device=device)

    def detect_ball(self, frame, conf=_DEFAULT_CONF_THRESHOLD, roi_corners=None):
        t0 = time.time()
        orig_h, orig_w = frame.shape[:2]

        self._update_frame_window(frame)
        stacked_chw = self._build_model_input()
        heatmap = np.asarray(self._model_fn(stacked_chw), dtype=np.float32)

        if self.show_performance_stats:
            print(f"TrackNet ball inference took {time.time() - t0:.2f} sec")

        peak_point, confidence = self._peak_from_heatmap(heatmap, conf)

        final_point = None
        if peak_point is not None:
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height
            scaled = (int(peak_point[0] * scale_x), int(peak_point[1] * scale_y))
            if self._point_in_roi(scaled, roi_corners):
                final_point = scaled

        self.last_detection = {
            "visible": final_point is not None,
            "accepted": False,
            "image": list(final_point) if final_point is not None else None,
            "confidence": confidence,
            "candidate_count": 1 if peak_point is not None else 0,
        }
        return list(final_point) if final_point is not None else [0, 0]

    def get_last_detection(self):
        return dict(self.last_detection)

    def clear(self):
        self._frame_window.clear()
        self.last_detection = self._empty_detection_state()

    def _update_frame_window(self, frame):
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        self._frame_window.append(resized)
        # 不足 3 帧（视频刚开始的前两次调用）：复制已知最早的一帧填充窗口
        # 前部，保证 stacked 输入 channel 数恒为 9。
        while len(self._frame_window) < _WINDOW_SIZE:
            self._frame_window.appendleft(self._frame_window[0])

    def _build_model_input(self):
        # deque 内部顺序是 oldest -> newest；上游 infer_on_video.py 用
        # `concatenate((img, img_prev, img_preprev), axis=2)`（newest 在前），
        # 这里反转对齐同一堆叠顺序。
        frames_newest_first = list(self._frame_window)[::-1]
        stacked = np.concatenate(frames_newest_first, axis=2).astype(np.float32) / 255.0
        return np.rollaxis(stacked, 2, 0)  # (9, H, W)

    def _peak_from_heatmap(self, heatmap, threshold):
        if heatmap is None or heatmap.size == 0:
            return None, None
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = float(heatmap[y_idx, x_idx])
        if confidence < threshold:
            return None, confidence
        return (int(x_idx), int(y_idx)), confidence

    def _point_in_roi(self, point, roi_corners):
        if roi_corners is None:
            return True
        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        padding = int(max(x2 - x1, y2 - y1) * self.roi_padding_ratio)
        return (x1 - padding) <= point[0] <= (x2 + padding) and (y1 - padding) <= point[1] <= (y2 + padding)

    def _empty_detection_state(self):
        return {
            "visible": False,
            "accepted": False,
            "image": None,
            "confidence": None,
            "candidate_count": 0,
        }


class _TorchBallTrackerNetAdapter:
    """把 torch nn.Module 包装成 (stacked_input[9,H,W]) -> heatmap[H,W] 的可
    调用对象，与测试注入的 fake model 遵循同一契约，使 TrackNetBallDetector
    的核心逻辑完全不关心 torch 细节（也就不需要在模块顶层 import torch）。
    """

    def __init__(self, model_path, device=None):
        import torch

        self._torch = torch
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        network_cls = _get_ball_tracker_net_class()
        self.model = network_cls(in_channels=9, out_channels=256).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[TrackNetBallDetector] 权重已加载 / weights loaded from {model_path} (device={self.device})")

    def __call__(self, stacked_input):
        torch = self._torch
        inp = torch.tensor(stacked_input).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(inp)  # (1, 256, H, W)
            # 逐像素在 256 个类别上取 argmax，重建灰度热图（类别下标即预测
            # 强度 0-255），归一化到 0-1 便于与默认阈值 0.5 直接比较。
            class_idx = out.argmax(dim=1)[0].float() / 255.0
        return class_idx.detach().cpu().numpy().astype(np.float32)


def _build_ball_tracker_net_class():
    """延迟构建网络结构类（内部 import torch，保证顶层模块不强依赖 torch）。

    结构逐字对应上游仓库 model.py 的 ConvBlock / BallTrackerNet
    （Apache-2.0，来源见文件头注释），仅将末尾 `.reshape(batch,
    out_channels, -1)` + 条件 softmax 省略（无参数的纯后处理变换，见模块
    docstring），其余卷积/池化/上采样层与权重形状逐一对应。
    """
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.block(x)

    class BallTrackerNet(nn.Module):
        def __init__(self, in_channels=9, out_channels=256):
            super().__init__()
            self.out_channels = out_channels

            self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64)
            self.conv2 = ConvBlock(in_channels=64, out_channels=64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = ConvBlock(in_channels=64, out_channels=128)
            self.conv4 = ConvBlock(in_channels=128, out_channels=128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv5 = ConvBlock(in_channels=128, out_channels=256)
            self.conv6 = ConvBlock(in_channels=256, out_channels=256)
            self.conv7 = ConvBlock(in_channels=256, out_channels=256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv8 = ConvBlock(in_channels=256, out_channels=512)
            self.conv9 = ConvBlock(in_channels=512, out_channels=512)
            self.conv10 = ConvBlock(in_channels=512, out_channels=512)
            self.ups1 = nn.Upsample(scale_factor=2)
            self.conv11 = ConvBlock(in_channels=512, out_channels=256)
            self.conv12 = ConvBlock(in_channels=256, out_channels=256)
            self.conv13 = ConvBlock(in_channels=256, out_channels=256)
            self.ups2 = nn.Upsample(scale_factor=2)
            self.conv14 = ConvBlock(in_channels=256, out_channels=128)
            self.conv15 = ConvBlock(in_channels=128, out_channels=128)
            self.ups3 = nn.Upsample(scale_factor=2)
            self.conv16 = ConvBlock(in_channels=128, out_channels=64)
            self.conv17 = ConvBlock(in_channels=64, out_channels=64)
            self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

            self._init_weights()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool3(x)
            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.ups1(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.ups2(x)
            x = self.conv14(x)
            x = self.conv15(x)
            x = self.ups3(x)
            x = self.conv16(x)
            x = self.conv17(x)
            x = self.conv18(x)
            return x

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.uniform_(module.weight, -0.05, 0.05)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    return BallTrackerNet


_ball_tracker_net_cls = None


def _get_ball_tracker_net_class():
    """惰性构建并缓存网络结构类，避免模块顶层无条件 import torch。"""
    global _ball_tracker_net_cls
    if _ball_tracker_net_cls is None:
        _ball_tracker_net_cls = _build_ball_tracker_net_class()
    return _ball_tracker_net_cls
