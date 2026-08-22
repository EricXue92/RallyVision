"""WASB-SBDT 网球检测封装（--ball-detector wasb 后端，Task 9b delta）。

上游仓库：[nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)（"Widely
Applicable Strong Baseline for Sports Ball Detection and Tracking"，
arXiv:2311.05237）。**该仓库根目录 `LICENSE.md` 明确写的是 MIT License**
（Copyright (c) 2023 NTT Communications Corporation）——与 Task 9 的
TrackNet/court keypoint 参考实现（yastrebksv 系，均无 LICENSE 文件）不同，
这里许可证状态是确定且宽松的，已实测核对原文，不是猜测。

网络结构：WASB 使用的骨干是仓库 `src/models/hrnet.py` 里的 `HRNet` 类
（改自 [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
`cls_hrnet.py`，该文件头部同样声明 MIT License，署名 Bin Xiao / Bowen
Cheng）。本文件下方 `_build_hrnet_class()` 是其结构的逐层抄录（conv/BN/
ReLU 拓扑、各 stage 的 branch/block 配置、fuse_layers 融合逻辑均与原文件
一致），仅有两处刻意改动，均已在函数内注释标注：
  1. `_make_deconv_layers` 原文用 `cfg.MODEL.EXTRA`（属性访问）取配置，
     这是因为上游用 hydra/OmegaConf 的 `DictConfig`（同时支持属性和下标
     访问）加载 yaml；本项目不引入 hydra/omegaconf 依赖，`cfg` 就是普通
     嵌套 dict，属性访问会 `AttributeError`，故改成 `cfg['MODEL']['EXTRA']`
     下标访问，语义不变（WASB 网球配置 `NUM_DECONVS=0`，这段代码本来就不
     会真正跑循环体，但赋值语句本身在函数入口无条件执行，不改会直接崩）。
  2. 顶层 `build_model()`/`HigherHRNet` 等仓库其它模型分支未抄入，只留
     tennis 权重实际用到的 HRNet 一支。

模型配置：`_HRNET_CFG` 逐字段对应仓库 `src/configs/model/wasb.yaml`
（`inp_width=512` `inp_height=288` `frames_in=frames_out=3`
`out_scales=[0]`，STAGE1-4 的 NUM_CHANNELS/NUM_BLOCKS/BLOCK 类型等）。已用
下载到的真实权重 `weights/wasb_tennis.pth` 验证：`model.load_state_dict(
state_dict, strict=True)` 零 missing/unexpected key，前向输出形状
`(1, 3, 288, 512)`，与配置数值吻合，不是臆造的结构。

权重来源：`MODEL_ZOO.md` 表格 "WASB (Ours)" 行 × "Tennis" 列的 Google
Drive 链接（file id `14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z`），见
`weights/README.md`。存放路径 `weights/wasb_tennis.pth`（gitignored，不
入库）。

输入约定（核对自仓库 `src/datasets/tennis.py` + `src/dataloaders/
dataset_loader.py`，不是猜测）：
  - 训练样本按 `frame_names[i:i+frames_in]` 取窗口，**列表顺序是时间正序
    （最早帧在前、最新帧在后）**，而 `imgs_t = torch.cat(imgs_t, dim=0)`
    直接按该顺序把每帧的 3 通道拼接——即 **channel 0-2 = 最早帧，channel
    6-8 = 最新帧**。这与 `tracknet_ball.py` 的 newest-first 堆叠顺序刚好
    相反（那是从 TrackNet 上游 `infer_on_video.py` 的 `concatenate((img,
    img_prev, img_preprev))` 反推的约定），两者互不兼容，不能混用同一个
    `_build_model_input`。
  - `frames_out=3` 与 `frames_in` 对齐（`annos = ball_xyvs[i:i+frames_in]`），
    即模型对窗口内**每一帧**都输出一张热图（channel 0=最早帧的球位置预测，
    channel 2=最新帧），我们只关心"当前帧"，取最后一个 channel（index
    `frames_out - 1`）。
  - 输入图像预处理是 `PIL.Image.open(...).convert('RGB')` + ImageNet
    `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`（见仓库
    `src/dataloaders/__init__.py::build_img_transforms`），**不是** TrackNet
    那种简单 `/255.0`。这里的差异不是简化风格问题——本仓库精确知道这份
    checkpoint 训练时喂的就是 RGB + ImageNet 归一化，跳过会实质性拉低精度，
    所以本文件对每帧做 `cv2.cvtColor(BGR2RGB)` + 归一化，而不是照抄
    TrackNet 的简化。

后处理：上游 `src/detectors/postprocessor.py::TracknetV2Postprocessor`
对模型输出先 `.sigmoid_()`，再用连通域（concomp）或 NMS 峰值抑制找多个
候选 blob。本项目沿用 Task 9 TrackNet 适配器建立的简化路线（模块 docstring
同款说明）：只取全局峰值 + 阈值判断，不做多候选/blob 检测——`_TorchWASBModelAdapter`
把 `sigmoid` + `argmax` 都做在 `__call__` 内部，对外仍是"喂 (9,H,W) 归一化
张量、吐 (H,W) 灰度热图"的同一契约，与注入测试用假模型的调用方式完全一致。

球员框 gating（Task 9b Step 1）复用 `tracknet_ball.py` 顶层的
`is_implausible_ball_jump` 纯函数，两后端共用同一套阈值常量，见该文件
docstring。
"""
import time
from collections import deque

import cv2
import numpy as np

from .tracknet_ball import is_implausible_ball_jump

# 上游 wasb.yaml 训练/推理输入分辨率（src/configs/model/wasb.yaml:
# inp_width/inp_height）
_MODEL_INPUT_WIDTH = 512
_MODEL_INPUT_HEIGHT = 288
_WINDOW_SIZE = 3
_DEFAULT_CONF_THRESHOLD = 0.5

# ImageNet 归一化参数（build_img_transforms 用的标准值），逐帧应用
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class WASBBallDetector:
    """WASB 球检测封装，接口与 `tracknet_ball.py::TrackNetBallDetector` /
    `detection/tennis_ball.py::TennisBallTracker` 完全对齐（同一套
    `detect_ball`/`last_detection`/`clear` 契约，可互相替换）。

    内部维护一个长度为 3 的滑窗（最近 3 帧，resize 到 512x288 + BGR→RGB +
    ImageNet 归一化后沿 channel 堆叠为 [9,H,W] 送入网络，**时间正序**——
    channel 0-2 最早帧、6-8 最新帧，见模块 docstring"输入约定"节，与
    TrackNetBallDetector 的 newest-first 顺序刻意不同）；不足 3 帧时（视频
    刚开始）复制已知最早的一帧填满窗口，与 TrackNetBallDetector 同一处理。

    支持注入假模型用于测试，无需 torch/权重：
        WASBBallDetector(model_path=None, model=fake_callable)
    其中 fake_callable(stacked_input: np.ndarray[9,H,W]) -> np.ndarray[H,W]
    （灰度热图，值域任意，内部做全局峰值 + 阈值判断；真实模型适配器
    `_TorchWASBModelAdapter` 把 torch 推理 + sigmoid + 取最新帧 channel 都
    封在内部，归一化到 0-1 后再喂给下游同一套峰值判断逻辑，对外暴露的调用
    契约与假模型完全一致）。
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
                "WASBBallDetector 需要 model_path（加载真实权重）或 model"
                "（注入可调用对象，通常用于测试）之一 / "
                "WASBBallDetector requires either model_path or an injected model callable"
            )

        self.input_width = input_width
        self.input_height = input_height
        self.roi_padding_ratio = roi_padding_ratio
        self.show_performance_stats = show_performance_stats

        self._frame_window = deque(maxlen=_WINDOW_SIZE)
        self.last_detection = self._empty_detection_state()
        self._last_accepted_point = None  # Task 9b gating：上一帧被接受的球点（跨帧持久）

        if model is not None:
            self._model_fn = model
        else:
            self._model_fn = _TorchWASBModelAdapter(model_path, device=device)

    def detect_ball(self, frame, conf=_DEFAULT_CONF_THRESHOLD, roi_corners=None, player_centers=None):
        t0 = time.time()
        orig_h, orig_w = frame.shape[:2]

        self._update_frame_window(frame)
        stacked_chw = self._build_model_input()
        heatmap = np.asarray(self._model_fn(stacked_chw), dtype=np.float32)

        if self.show_performance_stats:
            print(f"WASB ball inference took {time.time() - t0:.2f} sec")

        peak_point, raw_confidence = self._peak_from_heatmap(heatmap, conf)

        final_point = None
        if peak_point is not None:
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height
            scaled = (int(peak_point[0] * scale_x), int(peak_point[1] * scale_y))
            if self._point_in_roi(scaled, roi_corners):
                final_point = scaled

        # Task 9b Step 1：球员框 gating，与 TrackNetBallDetector 同一套纯函数、
        # 同一处理时机（ROI 通过之后）。
        if final_point is not None and is_implausible_ball_jump(
            final_point, self._last_accepted_point, player_centers, orig_w, orig_h
        ):
            final_point = None

        # 拒绝路径契约与 TrackNetBallDetector 对齐（tracknet_ball.py 同一处
        # 注释）：ROI/gating 任一环节拒绝，都视为该候选未通过候选提取阶段，
        # confidence/candidate_count 一并清零，不留「不可见但仍标着高置信度
        # 候选」的不一致状态。
        confidence = raw_confidence if final_point is not None else None
        candidate_count = 1 if final_point is not None else 0

        self.last_detection = {
            "visible": final_point is not None,
            "accepted": False,
            "image": list(final_point) if final_point is not None else None,
            "confidence": confidence,
            "candidate_count": candidate_count,
        }
        if final_point is not None:
            self._last_accepted_point = tuple(final_point)
        return list(final_point) if final_point is not None else [0, 0]

    def get_last_detection(self):
        return dict(self.last_detection)

    def clear(self):
        self._last_accepted_point = None
        self._frame_window.clear()
        self.last_detection = self._empty_detection_state()

    def _update_frame_window(self, frame):
        # BGR（cv2 读帧原始通道序）→RGB，对齐 checkpoint 训练时的 PIL RGB 输入
        # （见模块 docstring"输入约定"节）。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        self._frame_window.append(resized)
        while len(self._frame_window) < _WINDOW_SIZE:
            self._frame_window.appendleft(self._frame_window[0])

    def _build_model_input(self):
        # deque 内部顺序即 oldest -> newest，与上游训练时 `frame_names[i:i+
        # frames_in]` 的时间正序窗口一致，直接按该顺序堆叠（channel 0-2 最早
        # 帧、6-8 最新帧），不像 TrackNetBallDetector 那样需要反转。
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(_IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        normalized_frames = []
        for frame in self._frame_window:
            normalized = (frame.astype(np.float32) / 255.0 - mean) / std
            normalized_frames.append(normalized)
        stacked = np.concatenate(normalized_frames, axis=2)  # (H, W, 9)
        return np.rollaxis(stacked, 2, 0).astype(np.float32)  # (9, H, W)

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


class WASBBallTrackerAdapter:
    """把 WASBBallDetector 包成与 TrackNetBallTrackerAdapter 一致的
    detect_ball/update_trajectory/clear_trajectory 三件套接口，供 system.py
    `--ball-detector wasb` 时原地替换 `self.tennis_ball_tracker`（与
    tracknet_ball.py::TrackNetBallTrackerAdapter 同一套 wiring 模式，见该类
    docstring 的设计说明，此处不重复）。
    """

    def __init__(self, detector, trajectory_length=30):
        self._detector = detector
        self.tennis_ball_trajectory = deque(maxlen=trajectory_length)
        self.last_valid_position = None

    def detect_ball(self, frame, roi_corners=None, player_centers=None):
        return self._detector.detect_ball(frame, roi_corners=roi_corners, player_centers=player_centers)

    def update_trajectory(self, ball_position, roi_corners=None):
        if ball_position is None or list(ball_position) == [0, 0]:
            return [0, 0]
        point = [int(ball_position[0]), int(ball_position[1])]
        self.tennis_ball_trajectory.append(tuple(point))
        self.last_valid_position = tuple(point)
        return point

    def clear_trajectory(self):
        self.tennis_ball_trajectory.clear()
        self.last_valid_position = None
        self._detector.clear()

    def get_last_detection(self):
        return self._detector.get_last_detection()

    def get_trajectory(self):
        return list(self.tennis_ball_trajectory)

    def draw_trajectory(self, frame):
        if not self.tennis_ball_trajectory:
            return
        color = (87, 108, 255)
        points = list(self.tennis_ball_trajectory)
        for index, point in enumerate(points):
            radius = int(3 + (index / len(points)) * 4)
            cv2.circle(frame, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        latest_point = points[-1]
        cv2.circle(frame, latest_point, 6, (0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

    def handle_visualization(self, frame):
        if self.tennis_ball_trajectory:
            self.draw_trajectory(frame)


class _TorchWASBModelAdapter:
    """把 torch nn.Module（HRNet）包装成 (stacked_input[9,H,W]) ->
    heatmap[H,W] 的可调用对象，与测试注入的 fake model 遵循同一契约，使
    WASBBallDetector 的核心逻辑完全不关心 torch 细节（不需要在模块顶层
    import torch）。"""

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

        network_cls = _get_hrnet_class()
        self.model = network_cls(_HRNET_CFG).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        # 下载到的 weights/wasb_tennis.pth 是 {'model_state_dict': ...} 包装
        # 过的 checkpoint（已用真实文件核实过键名/形状，strict=True 零 missing/
        # unexpected）；兼容万一拿到未包装的裸 state_dict。
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[WASBBallDetector] 权重已加载 / weights loaded from {model_path} (device={self.device})")

    def __call__(self, stacked_input):
        torch = self._torch
        inp = torch.tensor(stacked_input).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(inp)  # {0: (1, frames_out=3, H, W)}
            logits = out[0]
            probs = torch.sigmoid(logits)
            # frames_out 的 channel 顺序与输入堆叠顺序一致（时间正序），最新
            # 帧是最后一个 channel（见模块 docstring"输入约定"节）。
            current_frame_heatmap = probs[0, -1]
        return current_frame_heatmap.detach().cpu().numpy().astype(np.float32)


def _build_hrnet_class():
    """延迟构建网络结构类（内部 import torch，保证顶层模块不强依赖 torch）。

    逐层对应上游 `src/models/hrnet.py::HRNet`（改自 HRNet-Image-
    Classification `cls_hrnet.py`，两者均 MIT License，见模块 docstring）。
    唯一改动是 `_make_deconv_layers` 里的属性访问改下标访问（同 docstring
    "网络结构"节说明的原因，不是行为改动——WASB 网球配置 NUM_DECONVS=0，
    这段循环体本来就不会真正执行）。
    """
    import torch.nn as nn

    BN_MOMENTUM = 0.1

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            return self.relu(out)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            return self.relu(out)

    class HighResolutionModule(nn.Module):
        def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
            super().__init__()
            self.num_inchannels = num_inchannels
            self.fuse_method = fuse_method
            self.num_branches = num_branches
            self.multi_scale_output = multi_scale_output
            self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU(True)

        def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
            downsample = None
            if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.num_inchannels[branch_index],
                        num_channels[branch_index] * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
                )
            layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
            self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
            for i in range(1, num_blocks[branch_index]):
                layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
            return nn.Sequential(*layers)

        def _make_branches(self, num_branches, block, num_blocks, num_channels):
            return nn.ModuleList(
                [self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)]
            )

        def _make_fuse_layers(self):
            if self.num_branches == 1:
                return None
            num_branches = self.num_branches
            num_inchannels = self.num_inchannels
            fuse_layers = []
            for i in range(num_branches if self.multi_scale_output else 1):
                fuse_layer = []
                for j in range(num_branches):
                    if j > i:
                        fuse_layer.append(
                            nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                                nn.BatchNorm2d(num_inchannels[i]),
                                nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                            )
                        )
                    elif j == i:
                        fuse_layer.append(None)
                    else:
                        conv3x3s = []
                        for k in range(i - j):
                            if k == i - j - 1:
                                num_outchannels_conv3x3 = num_inchannels[i]
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                        nn.BatchNorm2d(num_outchannels_conv3x3),
                                    )
                                )
                            else:
                                num_outchannels_conv3x3 = num_inchannels[j]
                                conv3x3s.append(
                                    nn.Sequential(
                                        nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                        nn.BatchNorm2d(num_outchannels_conv3x3),
                                        nn.ReLU(True),
                                    )
                                )
                        fuse_layer.append(nn.Sequential(*conv3x3s))
                fuse_layers.append(nn.ModuleList(fuse_layer))
            return nn.ModuleList(fuse_layers)

        def get_num_inchannels(self):
            return self.num_inchannels

        def forward(self, x):
            if self.num_branches == 1:
                return [self.branches[0](x[0])]
            for i in range(self.num_branches):
                x[i] = self.branches[i](x[i])
            x_fuse = []
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
            return x_fuse

    blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}

    class HRNet(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self._frames_in = cfg["frames_in"]
            self._frames_out = cfg["frames_out"]
            self._out_scales = cfg["out_scales"]
            self._stem_strides = cfg["MODEL"]["EXTRA"]["STEM"]["STRIDES"]
            self._stem_inplanes = cfg["MODEL"]["EXTRA"]["STEM"]["INPLANES"]

            self.conv1 = nn.Conv2d(
                3 * self._frames_in, self._stem_inplanes, kernel_size=3, stride=self._stem_strides[0], padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
            self.conv2 = nn.Conv2d(
                self._stem_inplanes, self._stem_inplanes, kernel_size=3, stride=self._stem_strides[1], padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)

            self.stage1_cfg = cfg["MODEL"]["EXTRA"]["STAGE1"]
            num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
            block = blocks_dict[self.stage1_cfg["BLOCK"]]
            num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
            self.layer1 = self._make_layer(block, self._stem_inplanes, num_channels, num_blocks)
            stage1_out_channel = block.expansion * num_channels

            self.stage2_cfg = cfg["MODEL"]["EXTRA"]["STAGE2"]
            num_channels = self.stage2_cfg["NUM_CHANNELS"]
            block = blocks_dict[self.stage2_cfg["BLOCK"]]
            num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
            self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
            self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

            self.stage3_cfg = cfg["MODEL"]["EXTRA"]["STAGE3"]
            num_channels = self.stage3_cfg["NUM_CHANNELS"]
            block = blocks_dict[self.stage3_cfg["BLOCK"]]
            num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
            self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
            self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

            self.stage4_cfg = cfg["MODEL"]["EXTRA"]["STAGE4"]
            num_channels = self.stage4_cfg["NUM_CHANNELS"]
            block = blocks_dict[self.stage4_cfg["BLOCK"]]
            num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
            self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
            self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

            self.num_deconvs = cfg["MODEL"]["EXTRA"]["DECONV"]["NUM_DECONVS"]
            self.deconv_config = cfg["MODEL"]["EXTRA"]["DECONV"]
            self.deconv_layers = self._make_deconv_layers(cfg, pre_stage_channels[0])
            self.final_layers = self._make_final_layers(cfg, pre_stage_channels)

        def _get_deconv_cfg(self, deconv_kernel):
            if deconv_kernel == 4:
                return deconv_kernel, 1, 0
            if deconv_kernel == 3:
                return deconv_kernel, 1, 1
            if deconv_kernel == 2:
                return deconv_kernel, 0, 0
            raise ValueError(f"unsupported deconv_kernel: {deconv_kernel}")

        def _make_final_layers(self, cfg, channels):
            kernel_size = cfg["MODEL"]["EXTRA"]["FINAL_CONV_KERNEL"]
            layers = []
            for scale in self._out_scales:
                layers.append(nn.Conv2d(in_channels=channels[scale], out_channels=self._frames_out, kernel_size=kernel_size))
            return nn.ModuleList(layers)

        def _make_deconv_layers(self, cfg, input_channels):
            # 原文这里用 `cfg.MODEL.EXTRA`（属性访问），改成下标访问的原因见
            # _build_hrnet_class 的 docstring。
            extra = cfg["MODEL"]["EXTRA"]
            deconv_cfg = extra["DECONV"]
            deconv_layers = []
            for i in range(deconv_cfg["NUM_DECONVS"]):
                output_channels = input_channels
                deconv_kernel, padding, output_padding = self._get_deconv_cfg(deconv_cfg["KERNEL_SIZE"][i])
                layers = [
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=input_channels,
                            out_channels=output_channels,
                            kernel_size=deconv_kernel,
                            stride=2,
                            padding=padding,
                            output_padding=output_padding,
                            bias=False,
                        ),
                        nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                ]
                deconv_layers.append(nn.Sequential(*layers))
                input_channels = output_channels
            return nn.ModuleList(deconv_layers)

        def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
            num_branches_cur = len(num_channels_cur_layer)
            num_branches_pre = len(num_channels_pre_layer)
            transition_layers = []
            for i in range(num_branches_cur):
                if i < num_branches_pre:
                    if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                        transition_layers.append(
                            nn.Sequential(
                                nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                                nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                            )
                        )
                    else:
                        transition_layers.append(None)
                else:
                    conv3x3s = []
                    for j in range(i + 1 - num_branches_pre):
                        inchannels = num_channels_pre_layer[-1]
                        outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                        conv3x3s.append(
                            nn.Sequential(
                                nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                            )
                        )
                    transition_layers.append(nn.Sequential(*conv3x3s))
            return nn.ModuleList(transition_layers)

        def _make_layer(self, block, inplanes, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
                )
            layers = [block(inplanes, planes, stride, downsample)]
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))
            return nn.Sequential(*layers)

        def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
            num_modules = layer_config["NUM_MODULES"]
            num_branches = layer_config["NUM_BRANCHES"]
            num_blocks = layer_config["NUM_BLOCKS"]
            num_channels = layer_config["NUM_CHANNELS"]
            block = blocks_dict[layer_config["BLOCK"]]
            fuse_method = layer_config["FUSE_METHOD"]
            modules = []
            for i in range(num_modules):
                reset_multi_scale_output = not (not multi_scale_output and i == num_modules - 1)
                modules.append(
                    HighResolutionModule(
                        num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output
                    )
                )
                num_inchannels = modules[-1].get_num_inchannels()
            return nn.Sequential(*modules), num_inchannels

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.layer1(x)

            x_list = []
            for i in range(self.stage2_cfg["NUM_BRANCHES"]):
                x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
            y_list = self.stage2(x_list)

            x_list = []
            for i in range(self.stage3_cfg["NUM_BRANCHES"]):
                x_list.append(self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i])
            y_list = self.stage3(x_list)

            x_list = []
            for i in range(self.stage4_cfg["NUM_BRANCHES"]):
                x_list.append(self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i])
            y_list = self.stage4(x_list)

            y_out = {}
            for scale in self._out_scales:
                x = y_list[scale]
                for i in range(self.num_deconvs):
                    x = self.deconv_layers[i][scale](x)
                y_out[scale] = self.final_layers[scale](x)
            return y_out

    return HRNet


# 逐字段对应 src/configs/model/wasb.yaml（见模块 docstring"模型配置"节）
_HRNET_CFG = {
    "frames_in": 3,
    "frames_out": 3,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": []},
        }
    },
}


_hrnet_cls = None


def _get_hrnet_class():
    """惰性构建并缓存网络结构类，避免模块顶层无条件 import torch。"""
    global _hrnet_cls
    if _hrnet_cls is None:
        _hrnet_cls = _build_hrnet_class()
    return _hrnet_cls
