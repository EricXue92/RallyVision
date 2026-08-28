"""TrackNetV4 网球检测封装（--ball-detector tracknetv4 后端）。

上游：https://github.com/TrackNetV4/TrackNetV4（**MIT License**，2025 —— 与本
项目其他几个上游不同，这份的使用/再分发权利是明确的）。论文思路：在
TrackNetV2 的编解码骨干末端接一层「运动提示注意力」（Motion Prompt），用相邻
帧灰度差经幂归一化得到注意力图，再逐帧调制输出热图，压低静止背景的假响应。

Upstream: github.com/TrackNetV4/TrackNetV4 (MIT). Motion-attention on top of a
TrackNetV2 encoder-decoder backbone.

与现有 `tracknet_ball.py`（yastrebksv/TrackNet 那支 TrackNetV2 变体）的差异，
逐条都会影响权重能否对上，改动前先读完：

1. **输入分辨率 512x288**（上游 constants.HEIGHT/WIDTH），不是 640x360。
2. **输入是 RGB、时间正序（旧->新）**，不是 BGR 逆序。9 个 channel 按
   `[帧t-2 RGB, 帧t-1 RGB, 帧t RGB]` 排列——Motion Prompt 层要按这个顺序做
   帧差，顺序错了注意力图就是反的。转换在 `_TorchTrackNetV4Adapter` 里做
   （见那里的 `_BGR_NEWEST_FIRST_TO_RGB_OLDEST_FIRST`），`TrackNetBallDetector`
   的取帧逻辑一行不用改。
3. **输出是 3 张 sigmoid 热图**（每个输入帧一张，值域 0-1），不是 256 类逐
   像素分类。取 index 2（最新帧）喂给下游同一套峰值逻辑。
4. **骨干带 3 条 skip 连接**（concat），`tracknet_ball.py` 那支没有——两边的
   网络结构不能互相复用。
5. **BatchNormalization 归一化维是 W 不是 C**。上游 Keras 代码写
   `BatchNormalization()`（默认 `axis=-1`）却把数据摆成 `channels_first`，于是
   实际归一化的是最后一维 = 宽。这是 TrackNetV2 系 Keras 实现的历史怪癖，
   权重形状（512/256/128/64，正好是各层的 W）已经证实。移植必须照搬，**不能
   换成 `nn.BatchNorm2d`**，否则权重形状对不上、对得上也是错的语义。
   见 `_WidthBatchNorm`。

权重来自上游在 new_tennis 数据集上训练的 checkpoint，用 `tools/convert_tracknetv4.py`
从 `.keras` 转成 `.pt`；下载与转换步骤见 weights/README.md。
"""
import numpy as np

from .tracknet_ball import TrackNetBallDetector

# 上游 constants.py：HEIGHT=288, WIDTH=512
MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 288

# Keras BatchNormalization 默认 epsilon（上游未改）
_BN_EPSILON = 1e-3

# 上游 MotionPromptLayer：输入按 (x*0.225+0.45) 还原到近似 [0,1] 后转灰度
_INPUT_RESCALE_SCALE = 0.225
_INPUT_RESCALE_SHIFT = 0.45
_GRAYSCALE_RGB_WEIGHTS = (0.299, 0.587, 0.114)

# 融合层类型：'A' = 论文 Type A（.keras 里叫 MotionIncorporationLayerV1），
# 'B' = Type B（V2），'none' = 不融合的 TrackNetV2 基线（对照组权重）
FUSION_TYPES = ("A", "B", "none")


def _build_tracknet_v4_class():
    """延迟构建网络结构类（内部 import torch，保持模块顶层不强依赖 torch）。"""
    import torch
    import torch.nn as nn

    class _WidthBatchNorm(nn.Module):
        """复刻 Keras `BatchNormalization(axis=-1)` 作用在 (B,C,H,W) 上的行为：
        归一化维是 **W**。参数/统计量长度 = 该层的宽，不是通道数。

        推理期恒用滑动统计量（等价 Keras 的 `training=False`），所以这里不需要
        区分 train/eval——本项目只做推理。
        """

        def __init__(self, width, eps=_BN_EPSILON):
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(width))
            self.beta = nn.Parameter(torch.zeros(width))
            self.register_buffer("moving_mean", torch.zeros(width))
            self.register_buffer("moving_variance", torch.ones(width))

        def forward(self, x):
            normalized = (x - self.moving_mean) * torch.rsqrt(self.moving_variance + self.eps)
            return normalized * self.gamma + self.beta

    class _MotionPrompt(nn.Module):
        """上游 MotionPromptLayer：相邻帧灰度差 -> 幂归一化 -> 注意力图 (B,2,H,W)。

        两个可训练标量 a、b 控制幂归一化的陡峭度与偏置（见 `power_normalization`）。
        """

        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor(0.1))
            self.b = nn.Parameter(torch.tensor(0.0))
            # persistent=False：灰度权重是常量不是学到的参数，不进 state_dict，
            # 免得转换脚本那边多出一个搬不出来的 key
            self.register_buffer(
                "gray_weights", torch.tensor(_GRAYSCALE_RGB_WEIGHTS).view(1, 1, 3, 1, 1),
                persistent=False,
            )

        def forward(self, x):
            # x: (B, 9, H, W) -> (B, T=3, C=3, H, W)
            batch, _, height, width = x.shape
            sequence = x.view(batch, 3, 3, height, width)
            sequence = sequence * _INPUT_RESCALE_SCALE + _INPUT_RESCALE_SHIFT
            gray = (sequence * self.gray_weights).sum(dim=2)  # (B, 3, H, W)
            frame_diff = gray[:, 1:] - gray[:, :-1]  # (B, 2, H, W)
            # power_normalization: 1 / (1 + exp(-(5/(0.45*|tanh(a)|+0.1)) * (|diff| - 0.6*tanh(b))))
            steepness = 5.0 / (0.45 * torch.abs(torch.tanh(self.a)) + 1e-1)
            return torch.sigmoid(steepness * (torch.abs(frame_diff) - 0.6 * torch.tanh(self.b)))

    class TrackNetV4Net(nn.Module):
        """TrackNetV2 骨干（带 skip）+ 可选 Motion Prompt 融合。

        层的编号与上游 Keras 图一一对应（conv0..conv17 / bn0..bn16），
        `tools/convert_tracknetv4.py` 依赖这个命名做权重搬运，别重命名。
        """

        def __init__(self, fusion="A", input_height=MODEL_INPUT_HEIGHT, input_width=MODEL_INPUT_WIDTH):
            super().__init__()
            if fusion not in FUSION_TYPES:
                raise ValueError(f"fusion 必须是 {FUSION_TYPES} 之一，收到 {fusion!r}")
            self.fusion = fusion

            # 各层的 W（BN 归一化维）随 3 次下采样 / 3 次上采样变化
            w_full, w_2, w_4, w_8 = (
                input_width,
                input_width // 2,
                input_width // 4,
                input_width // 8,
            )

            def conv(in_ch, out_ch, kernel_size=3):
                padding = kernel_size // 2
                return nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding, bias=True)

            channels = [
                (9, 64, w_full), (64, 64, w_full),          # conv0, conv1   -> skip1
                (64, 128, w_2), (128, 128, w_2),            # conv2, conv3   -> skip2
                (128, 256, w_4), (256, 256, w_4), (256, 256, w_4),   # conv4-6 -> skip3
                (256, 512, w_8), (512, 512, w_8), (512, 512, w_8),   # conv7-9
                (768, 256, w_4), (256, 256, w_4), (256, 256, w_4),   # conv10-12（concat skip3）
                (384, 128, w_2), (128, 128, w_2),           # conv13-14（concat skip2）
                (192, 64, w_full), (64, 64, w_full),        # conv15-16（concat skip1）
            ]
            for index, (in_ch, out_ch, width) in enumerate(channels):
                setattr(self, f"conv{index}", conv(in_ch, out_ch))
                setattr(self, f"bn{index}", _WidthBatchNorm(width))
            # 末端 1x1 卷积出 3 通道热图，后面直接接融合 + sigmoid，无 BN
            self.conv17 = conv(64, 3, kernel_size=1)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.motion = _MotionPrompt() if fusion in ("A", "B") else None

        def _block(self, index, x):
            return getattr(self, f"bn{index}")(torch.relu(getattr(self, f"conv{index}")(x)))

        def forward(self, x):
            attention = self.motion(x) if self.motion is not None else None

            x = self._block(0, x)
            skip1 = self._block(1, x)

            x = self._block(2, self.pool(skip1))
            skip2 = self._block(3, x)

            x = self._block(4, self.pool(skip2))
            x = self._block(5, x)
            skip3 = self._block(6, x)

            x = self._block(7, self.pool(skip3))
            x = self._block(8, x)
            x = self._block(9, x)

            x = torch.cat([self.upsample(x), skip3], dim=1)
            x = self._block(10, x)
            x = self._block(11, x)
            x = self._block(12, x)

            x = torch.cat([self.upsample(x), skip2], dim=1)
            x = self._block(13, x)
            x = self._block(14, x)

            x = torch.cat([self.upsample(x), skip1], dim=1)
            x = self._block(15, x)
            x = self._block(16, x)

            feature = self.conv17(x)  # (B, 3, H, W)，每个输入帧一张

            if attention is not None:
                if self.fusion == "A":
                    # Type A：首帧不调制（它没有「前一帧」可做差）
                    fused = torch.stack(
                        [
                            feature[:, 0],
                            feature[:, 1] * attention[:, 0],
                            feature[:, 2] * attention[:, 1],
                        ],
                        dim=1,
                    )
                else:
                    # Type B：首帧也调制，中间帧取两张注意力图的均值
                    fused = torch.stack(
                        [
                            feature[:, 0] * attention[:, 0],
                            feature[:, 1] * ((attention[:, 0] + attention[:, 1]) / 2.0),
                            feature[:, 2] * attention[:, 1],
                        ],
                        dim=1,
                    )
                feature = fused

            return torch.sigmoid(feature)

    return TrackNetV4Net


_tracknet_v4_cls = None


def get_tracknet_v4_class():
    """惰性构建并缓存网络结构类。"""
    global _tracknet_v4_cls
    if _tracknet_v4_cls is None:
        _tracknet_v4_cls = _build_tracknet_v4_class()
    return _tracknet_v4_cls


# `TrackNetBallDetector` 堆出来的 9 通道是「BGR、时间逆序（新->旧）」
# （见 tracknet_ball.py::_build_model_input，那是 yastrebksv 那支的口径）。
# TrackNetV4 要的是「RGB、时间正序（旧->新）」，两者恰好是整段逆序：
#   [新B,新G,新R, 中B,中G,中R, 旧B,旧G,旧R]  --逆序-->
#   [旧R,旧G,旧B, 中R,中G,中B, 新R,新G,新B]
# 所以只需一次 channel 反转，取帧那侧一行不用改。
_BGR_NEWEST_FIRST_TO_RGB_OLDEST_FIRST = [8, 7, 6, 5, 4, 3, 2, 1, 0]

# 3 张输出热图对应 3 个输入帧，正序下 index 2 = 最新帧（即当前要检测的这帧）
_NEWEST_FRAME_OUTPUT_INDEX = 2

# 峰值亚像素精化：sigmoid 热图在球心附近会饱和成一小片 ~1.0 的区域，
# 直接 argmax 会在片内任意抖动（512x288 上 ±2px，映回 1280x720 就是 ±5px，
# 落点精度吃得下这个误差就白改了）。改成在峰值邻域内取强度加权质心。
_CENTROID_WINDOW_HALF = 6
_CENTROID_RELATIVE_FLOOR = 0.5  # 邻域内低于峰值一半的像素不参与加权


class _TorchTrackNetV4Adapter:
    """把 TrackNetV4 包装成 `(stacked_input[9,H,W]) -> heatmap[H,W]` 可调用对象，
    与 `tracknet_ball.py::_TorchBallTrackerNetAdapter` 遵循同一契约（也就同样
    支持在测试里换成 fake callable，核心逻辑不碰 torch）。
    """

    def __init__(self, model_path, fusion="A", device=None, input_height=MODEL_INPUT_HEIGHT,
                 input_width=MODEL_INPUT_WIDTH):
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

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        # 转换脚本会把 fusion 类型写进 checkpoint；显式传参优先（便于实验）
        state_dict = checkpoint.get("state_dict", checkpoint)
        checkpoint_fusion = checkpoint.get("fusion") if isinstance(checkpoint, dict) else None
        self.fusion = fusion or checkpoint_fusion or "A"

        network_cls = get_tracknet_v4_class()
        self.model = network_cls(fusion=self.fusion, input_height=input_height, input_width=input_width)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        print(
            f"[TrackNetV4BallDetector] 权重已加载 / weights loaded from {model_path} "
            f"(fusion={self.fusion}, device={self.device})"
        )

    def __call__(self, stacked_input):
        torch = self._torch
        reordered = np.ascontiguousarray(stacked_input[_BGR_NEWEST_FIRST_TO_RGB_OLDEST_FIRST])
        inp = torch.from_numpy(reordered).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(inp)  # (1, 3, H, W)，已过 sigmoid
            heatmap = out[0, _NEWEST_FRAME_OUTPUT_INDEX]
        return heatmap.detach().cpu().numpy().astype(np.float32)


class TrackNetV4BallDetector(TrackNetBallDetector):
    """TrackNetV4 版检测器。

    滑窗取帧、远场 ROI 二次推理、球场 ROI 过滤、球员 gating 全部继承基类不变，
    只换两处：模型适配器（分辨率/通道序/输出语义）和峰值提取（加权质心精化）。
    """

    def __init__(self, model_path=None, model=None, fusion="A", device=None,
                 input_width=MODEL_INPUT_WIDTH, input_height=MODEL_INPUT_HEIGHT, **kwargs):
        if model is None and model_path is None:
            raise ValueError(
                "TrackNetV4BallDetector 需要 model_path（加载真实权重）或 model"
                "（注入可调用对象，通常用于测试）之一 / "
                "TrackNetV4BallDetector requires either model_path or an injected model callable"
            )
        if model is None:
            model = _TorchTrackNetV4Adapter(
                model_path, fusion=fusion, device=device,
                input_height=input_height, input_width=input_width,
            )
        super().__init__(
            model=model, input_width=input_width, input_height=input_height, **kwargs
        )

    def _peak_from_heatmap(self, heatmap, threshold):
        """峰值 + 邻域强度加权质心（返回浮点坐标，基类会按缩放比换算回帧坐标）。"""
        if heatmap is None or heatmap.size == 0:
            return None, None
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = float(heatmap[y_idx, x_idx])
        if confidence < threshold:
            return None, confidence

        height, width = heatmap.shape
        y0 = max(0, y_idx - _CENTROID_WINDOW_HALF)
        y1 = min(height, y_idx + _CENTROID_WINDOW_HALF + 1)
        x0 = max(0, x_idx - _CENTROID_WINDOW_HALF)
        x1 = min(width, x_idx + _CENTROID_WINDOW_HALF + 1)
        window = heatmap[y0:y1, x0:x1]
        weights = np.where(window >= confidence * _CENTROID_RELATIVE_FLOOR, window, 0.0)
        total = float(weights.sum())
        if total <= 0:  # 理论上不会（峰值自己就 >= floor），防御性回退到 argmax
            return (int(x_idx), int(y_idx)), confidence

        ys, xs = np.nonzero(weights)
        centroid_x = float((weights[ys, xs] * (xs + x0)).sum() / total)
        centroid_y = float((weights[ys, xs] * (ys + y0)).sum() / total)
        return (centroid_x, centroid_y), confidence
