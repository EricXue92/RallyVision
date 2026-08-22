"""球场 14 关键点检测封装。

模型：yastrebksv/TennisCourtDetector（Apache-2.0）
https://github.com/yastrebksv/TennisCourtDetector
权重下载见 weights/README.md；网络结构 (BallTrackerNet) 直接抄自该仓库
的 tracknet.py（Apache-2.0，见下方类注释）。

点序映射（上游模型输出通道 -> 本项目 COURT_KEYPOINTS_M 索引）
================================================================
上游 `court_reference.py` 里 `CourtReference.key_points` 的构造顺序：

    key_points = [*baseline_top, *baseline_bottom,
                  *left_inner_line, *right_inner_line,
                  *top_inner_line, *bottom_inner_line,
                  *middle_line]

注意 `*` 是把每个二元组**依次完整展开**（不是与下一条线交替展开），
即 `left_inner_line=(p_top, p_bottom)` 会连续贡献两个点，然后才轮到
`right_inner_line`。按其像素参考系坐标展开（court_width=1117px ≈
10.97m 双打宽，court_height=2408px ≈ 23.77m 场长，561px 顶边线 /
2935px 底边线 / 423px & 1242px 单打边线 / 1110px & 2386px 发球线 /
832px 中线）：

| 上游索引 | 上游点（几何含义）                | 本项目 COURT_KEYPOINTS_M 索引 | 世界坐标 (x, y) 米 |
|---------|----------------------------------|-------------------------------|--------------------|
| 0       | baseline_top[0]    左上角（双打）  | 0                              | (0, 0)             |
| 1       | baseline_top[1]    右上角（双打）  | 1                              | (10.97, 0)         |
| 2       | baseline_bottom[0] 左下角（双打）  | 2                              | (0, 23.77)         |
| 3       | baseline_bottom[1] 右下角（双打）  | 3                              | (10.97, 23.77)     |
| 4       | left_inner_line[0]  顶边线单打左   | 4                              | (1.37, 0)          |
| 5       | left_inner_line[1]  底边线单打左   | 6                              | (1.37, 23.77)      |
| 6       | right_inner_line[0] 顶边线单打右   | 5                              | (9.60, 0)          |
| 7       | right_inner_line[1] 底边线单打右   | 7                              | (9.60, 23.77)      |
| 8       | top_inner_line[0]    顶发球线左    | 8                              | (1.37, 5.485)      |
| 9       | top_inner_line[1]    顶发球线右    | 9                              | (9.60, 5.485)      |
| 10      | bottom_inner_line[0] 底发球线左    | 10                             | (1.37, 18.285)     |
| 11      | bottom_inner_line[1] 底发球线右    | 11                             | (9.60, 18.285)     |
| 12      | middle_line[0]  顶 T 点            | 12                             | (5.485, 5.485)     |
| 13      | middle_line[1]  底 T 点            | 13                             | (5.485, 18.285)    |

即除索引 5/6（`left_inner_line[1]` 与 `right_inner_line[0]`）互换外，
其余全部恒等。**这一处互换最初分析源码时被漏看**（误以为
`*left_inner_line, *right_inner_line` 是逐点交替展开），导致 Step 4
冒烟测试第一次跑出 ~140px 的巨大重投影误差；通过肉眼核对
`outputs/kp_smoke.png` 上 idx5/idx6 的像素位置各自出现在了物理上相反的
行（idx5 落在底边线行、idx6 落在顶边线行），定位并修正为下表映射，
修正后重投影误差降到个位数（见 task-4-report.md）。
"""
import numpy as np

# 上游模型第 i 个输出通道 -> 本项目 COURT_KEYPOINTS_M 的第 j 行索引。
# 见上方模块 docstring 表格；索引 5/6 互换，其余恒等。
_UPSTREAM_TO_CANONICAL_INDEX = [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13]

# 标准场 14 点 (x, y)，米。与 tests/test_camera.py 的 COURT_POINTS 数值一致
# （该文件已改为直接 import 这份，删除重复字面量，见 Task 4 Step 3）。
COURT_KEYPOINTS_M = np.array([
    [0, 0], [10.97, 0], [0, 23.77], [10.97, 23.77],
    [1.37, 0], [9.60, 0], [1.37, 23.77], [9.60, 23.77],
    [1.37, 5.485], [9.60, 5.485], [1.37, 18.285], [9.60, 18.285],
    [5.485, 5.485], [5.485, 18.285],
], dtype=float)

# 上游训练/推理输入分辨率
_MODEL_INPUT_WIDTH = 640
_MODEL_INPUT_HEIGHT = 360
_CONF_THRESHOLD = 0.5
_MIN_VALID_POINTS = 6


class CourtKeypointDetector:
    """封装 yastrebksv/TennisCourtDetector 的 14 点热图网络。

    ``BallTrackerNet`` 网络结构逐字复制自上游仓库 tracknet.py
    （https://github.com/yastrebksv/TennisCourtDetector/blob/main/tracknet.py，
    Apache-2.0 许可），仅保留结构定义，不含其训练代码。
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
        self.model = network_cls(out_channels=15).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[CourtKeypointDetector] 权重已加载 / weights loaded from {model_path} (device={self.device})")

    def detect(self, frame):
        """检测球场 14 关键点。

        Args:
            frame: BGR 图像 (H, W, 3)，任意分辨率。

        Returns:
            np.ndarray[14, 2] 像素坐标（按 COURT_KEYPOINTS_M 顺序排列），
            置信不足的点为 NaN；有效点 < 6 时返回 None。
        """
        import cv2

        torch = self._torch
        orig_h, orig_w = frame.shape[:2]
        img = cv2.resize(frame, (_MODEL_INPUT_WIDTH, _MODEL_INPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.0)
        inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(inp)[0]
            heatmaps = torch.sigmoid(out).cpu().numpy()

        scale_x = orig_w / _MODEL_INPUT_WIDTH
        scale_y = orig_h / _MODEL_INPUT_HEIGHT

        raw_points = np.full((14, 2), np.nan, dtype=float)
        valid_count = 0
        for upstream_idx in range(14):
            heatmap = heatmaps[upstream_idx]
            conf = float(heatmap.max())
            canonical_idx = _UPSTREAM_TO_CANONICAL_INDEX[upstream_idx]
            if conf < _CONF_THRESHOLD:
                continue
            y_pred, x_pred = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            raw_points[canonical_idx] = [x_pred * scale_x, y_pred * scale_y]
            valid_count += 1

        if valid_count < _MIN_VALID_POINTS:
            return None
        return raw_points


def _build_ball_tracker_net_class():
    """延迟构建网络结构类（内部 import torch，保证顶层模块不强依赖 torch）。

    结构逐字对应上游仓库 tracknet.py 的 ConvBlock / BallTrackerNet
    （Apache-2.0，来源见文件头注释），未做任何改动。
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
        def __init__(self, out_channels=14):
            super().__init__()
            self.out_channels = out_channels

            self.conv1 = ConvBlock(in_channels=3, out_channels=64)
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
