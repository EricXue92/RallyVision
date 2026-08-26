"""TrackNet 网球检测封装（--ball-detector tracknet 后端）。

模型架构依据 TrackNet 论文（Huang et al. 2019, arXiv:1907.03698）；参考实现：
https://github.com/yastrebksv/TrackNet（该仓库未声明许可证——GitHub API
`license` 字段为 None，仓库根目录也没有 LICENSE 文件，使用/再分发权利未经
核实，不要当作 Apache-2.0 或其他任何已知许可证对待）。
Architecture per the TrackNet paper (Huang et al. 2019); reference
implementation: github.com/yastrebksv/TrackNet (no license declared
upstream — usage/redistribution status unverified).
权重下载见 weights/README.md；网络结构 (BallTrackerNet) 抄自该参考实现的
model.py，仅保留结构定义，不含训练代码。

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

# Task 9b Step 1: 球员框 gating 常数（TrackNet / WASB 两后端共用）。
# 阈值以 1280x720 为基准分辨率标定，按实际帧分辨率等比缩放（见 _gating_scale）。
_GATING_REFERENCE_WIDTH = 1280
_GATING_REFERENCE_HEIGHT = 720
_GATING_JUMP_THRESHOLD_PX = 150  # 候选点距上一帧接受点超过此距离才算「跳变」
_GATING_PLAYER_PROXIMITY_THRESHOLD_PX = 300  # 跳变点距最近球员中心在此距离内视为合法击球


def _gating_resolution_scale(frame_width, frame_height):
    """按帧分辨率相对 1280x720 基准的等比缩放系数（宽高各自比例取平均，
    兼容非 16:9 输入）。frame_width/frame_height 缺失或为 0 时返回 1.0
    （等同不缩放，只在真拿不到分辨率时退化，正常调用路径恒有值）。"""
    if not frame_width or not frame_height:
        return 1.0
    return ((frame_width / _GATING_REFERENCE_WIDTH) + (frame_height / _GATING_REFERENCE_HEIGHT)) / 2.0


def _euclidean_distance(point_a, point_b):
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


# 远场 ROI 二次推理（CourtCheck infer_with_far_roi 思想，代码全部重写——上游
# 无 License，只移植算法思路）：整帧压到 640x360 后远半场的球只剩 1~2 像素，
# 是发球/深球检测洞的主因。远场矩形 = 远半场四点（远端两底线角 + 球网两端
# 的图像坐标）bbox 加 padding：左右各 10% bbox 宽；上方 35% bbox 高（球飞行
# 弧线与底线后落点都在底线上方）；下方 10% bbox 高。
_FAR_ROI_HORIZONTAL_PAD_RATIO = 0.10
_FAR_ROI_TOP_PAD_RATIO = 0.35
_FAR_ROI_BOTTOM_PAD_RATIO = 0.10
_FAR_ROI_MIN_POINTS = 3


def compute_far_roi_rect(image_points, frame_width, frame_height):
    """远半场图像点集 -> 裁剪矩形 (x1, y1, x2, y2)（int，已 clamp 到帧内）。

    image_points 通常是 [远端左底角, 远端右底角, 网左端, 网右端] 的图像坐标。
    含 NaN 的点被剔除；有效点 < 3、bbox 零宽/零高、或 clamp 后矩形无面积时
    返回 None（调用方视为不启用远场二次推理，行为退回单次全帧推理）。
    """
    points = [
        (float(p[0]), float(p[1]))
        for p in image_points
        if p is not None and len(p) == 2 and np.isfinite(p[0]) and np.isfinite(p[1])
    ]
    if len(points) < _FAR_ROI_MIN_POINTS:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    box_w = max(xs) - min(xs)
    box_h = max(ys) - min(ys)
    if box_w <= 0 or box_h <= 0:
        return None
    x1 = int(round(min(xs) - box_w * _FAR_ROI_HORIZONTAL_PAD_RATIO))
    x2 = int(round(max(xs) + box_w * _FAR_ROI_HORIZONTAL_PAD_RATIO))
    y1 = int(round(min(ys) - box_h * _FAR_ROI_TOP_PAD_RATIO))
    y2 = int(round(max(ys) + box_h * _FAR_ROI_BOTTOM_PAD_RATIO))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(int(frame_width), x2), min(int(frame_height), y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1, y1, x2, y2)


def _point_in_rect(point, rect):
    x1, y1, x2, y2 = rect
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def is_implausible_ball_jump(
    candidate_point,
    last_accepted_point,
    player_centers,
    frame_width=_GATING_REFERENCE_WIDTH,
    frame_height=_GATING_REFERENCE_HEIGHT,
):
    """球员框 gating 纯函数（Task 9b Step 1，TrackNet / WASB 两后端共用，
    供两边 detect_ball 内部 import 调用，也单独可测）。

    球的合法位置突变只发生在球员击球处：候选球点若同时满足「距上一帧接受点
    > 150px」且「距 player_centers 中每一个球员中心都 > 300px」，判定为误检，
    返回 True（应丢弃该候选，detect_ball 视为本帧不可见）。否则返回 False
    （放行）。两个阈值都按 frame_width/frame_height 相对 1280x720 基准等比
    缩放（见 _gating_resolution_scale）。

    - last_accepted_point 为 None（无上一帧参照，如回合刚开始/首次检测）→
      不构成「跳变」判定的基础，直接放行（False）。
    - player_centers 为 None 或空 → 拿不到球员数据时 gating 直接放行
      （False），不允许因缺球员检测数据而误丢球（Task 9b brief 明确要求）。
    - player_centers 中的 None 元素（该侧球员本帧未检出）会被跳过，不参与
      距离比较。
    """
    if last_accepted_point is None:
        return False
    if not player_centers:
        return False

    scale = _gating_resolution_scale(frame_width, frame_height)
    jump_threshold = _GATING_JUMP_THRESHOLD_PX * scale
    proximity_threshold = _GATING_PLAYER_PROXIMITY_THRESHOLD_PX * scale

    if _euclidean_distance(candidate_point, last_accepted_point) <= jump_threshold:
        return False  # 不算跳变，正常连续轨迹，放行

    for center in player_centers:
        if center is None:
            continue
        if _euclidean_distance(candidate_point, center) <= proximity_threshold:
            return False  # 跳变但靠近球员（合法击球点），放行

    return True  # 跳变且远离所有球员，判误检丢弃


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
        # 远场 ROI 二次推理的独立滑窗（裁剪帧）；矩形变化（重标定）时重置，
        # 避免窗口内裁剪几何不一致
        self._far_frame_window = deque(maxlen=_WINDOW_SIZE)
        self._last_far_rect = None
        self.last_detection = self._empty_detection_state()
        self._last_accepted_point = None  # Task 9b gating：上一帧被接受的球点（跨帧持久）

        if model is not None:
            self._model_fn = model
        else:
            self._model_fn = _TorchBallTrackerNetAdapter(model_path, device=device)

    def detect_ball(self, frame, conf=_DEFAULT_CONF_THRESHOLD, roi_corners=None, player_centers=None, far_roi_rect=None):
        t0 = time.time()
        orig_h, orig_w = frame.shape[:2]

        self._update_frame_window(frame)
        stacked_chw = self._build_model_input()
        heatmap = np.asarray(self._model_fn(stacked_chw), dtype=np.float32)

        if self.show_performance_stats:
            print(f"TrackNet ball inference took {time.time() - t0:.2f} sec")

        peak_point, raw_confidence = self._peak_from_heatmap(heatmap, conf)

        full_point = None
        if peak_point is not None:
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height
            full_point = (int(peak_point[0] * scale_x), int(peak_point[1] * scale_y))

        # 远场 ROI 二次推理与融合：全帧漏检时用远场结果补洞；两边都检到且全帧
        # 结果落在远场矩形内时优先远场结果（裁剪后有效分辨率更高，坐标更准）。
        # 融合发生在 ROI 过滤 / 球员 gating 之前——两道守卫统一作用于融合结果。
        candidate, candidate_confidence = full_point, raw_confidence
        if far_roi_rect is not None:
            far_point, far_confidence = self._detect_in_far_roi(frame, far_roi_rect, conf)
            if far_point is not None and (full_point is None or _point_in_rect(full_point, far_roi_rect)):
                candidate, candidate_confidence = far_point, far_confidence

        final_point = None
        raw_confidence = candidate_confidence
        if candidate is not None and self._point_in_roi(candidate, roi_corners):
            final_point = candidate

        # Task 9b Step 1：球员框 gating——ROI 通过之后再过一道「跳变且远离
        # 所有球员」的误检过滤（is_implausible_ball_jump 是纯函数，player_centers
        # 拿不到时直接放行，不因缺球员数据丢球）。gating 拒绝与 ROI 拒绝走同一套
        # 「视为该候选未通过候选提取阶段」口径，下面 confidence/candidate_count
        # 清零逻辑统一处理，不需要在这里分别处理。
        if final_point is not None and is_implausible_ball_jump(
            final_point, self._last_accepted_point, player_centers, orig_w, orig_h
        ):
            final_point = None

        # 与 TennisBallTracker 对齐（tennis_ball.py:70-77）：它的 `candidates`
        # 由 `_extract_candidates` 产出，ROI 过滤在候选提取阶段就做了
        # （`_point_in_roi` 调用见其 _extract_candidates 内部），所以
        # `candidate_count = len(candidates)` 天然已经是「过 ROI 之后」的计数，
        # `confidence`/`image` 只在 selected 非空（=candidates 非空）时才非
        # None——即 visible / confidence 非 None / candidate_count>0 三者恒
        # 同步，没有「ROI 拒绝但 candidate_count 仍 >0」这种状态。本检测器
        # 只有一个全局热图峰值候选，按同一口径:峰值过阈值但被 ROI 拒绝时，
        # 视为该候选未通过候选提取阶段，一并清零 confidence/candidate_count。
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
        self._far_frame_window.clear()
        self._last_far_rect = None
        self.last_detection = self._empty_detection_state()

    def _detect_in_far_roi(self, frame, far_rect, conf):
        """远场裁剪二次推理：裁剪 -> resize 到模型输入分辨率 -> 推理 -> 峰值
        坐标按裁剪矩形回映射到帧坐标。返回 (point 或 None, confidence 或 None)。"""
        if far_rect != self._last_far_rect:
            self._far_frame_window.clear()
            self._last_far_rect = far_rect
        x1, y1, x2, y2 = far_rect
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        resized = cv2.resize(crop, (self.input_width, self.input_height))
        self._far_frame_window.append(resized)
        while len(self._far_frame_window) < _WINDOW_SIZE:
            self._far_frame_window.appendleft(self._far_frame_window[0])

        frames_newest_first = list(self._far_frame_window)[::-1]
        stacked = np.concatenate(frames_newest_first, axis=2).astype(np.float32) / 255.0
        heatmap = np.asarray(self._model_fn(np.rollaxis(stacked, 2, 0)), dtype=np.float32)
        peak_point, confidence = self._peak_from_heatmap(heatmap, conf)
        if peak_point is None:
            return None, confidence
        scale_x = (x2 - x1) / self.input_width
        scale_y = (y2 - y1) / self.input_height
        return (int(x1 + peak_point[0] * scale_x), int(y1 + peak_point[1] * scale_y)), confidence

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


class TrackNetBallTrackerAdapter:
    """把 TrackNetBallDetector 包成与 TennisBallTracker 相同的 detect_ball/update_trajectory/
    clear_trajectory 三件套接口，供 system.py `--ball-detector tracknet` 时原地替换
    `self.tennis_ball_tracker`，不改 `_process_frame` 里的调用方式（Task 10 wiring）。

    TrackNetBallDetector.detect_ball 内部已做置信度阈值 + ROI 过滤（检测阶段守卫，见其
    docstring），语义上已相当于 TennisBallTracker 的 detect_ball+update_trajectory 两步合一
    ——ROI 拒绝时直接返回 [0,0]。这里的 update_trajectory 因此只做与 TennisBallTracker 语义
    一致的直通："[0,0]即缺测"，不重复实现 YOLO 多候选框才需要的跳变外点竞争剔除
    （TrackNet 单峰值热图检测没有多候选框互相打分的场景）。trajectory 缓冲区仅用于保持
    get_trajectory()/draw_trajectory() 等下游可视化接口不崩，不影响检测正确性。
    """

    def __init__(self, detector, trajectory_length=30):
        self._detector = detector
        self.tennis_ball_trajectory = deque(maxlen=trajectory_length)
        self.last_valid_position = None

    def detect_ball(self, frame, roi_corners=None, player_centers=None, far_roi_rect=None):
        return self._detector.detect_ball(
            frame, roi_corners=roi_corners, player_centers=player_centers, far_roi_rect=far_roi_rect
        )

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

    结构逐字对应参考实现 model.py 的 ConvBlock / BallTrackerNet（来源 +
    许可证状态见文件头模块 docstring——上游未声明许可证，不是 Apache-2.0），
    仅将末尾 `.reshape(batch, out_channels, -1)` + 条件 softmax 省略（无参数
    的纯后处理变换，见模块 docstring），其余卷积/池化/上采样层与权重形状
    逐一对应。
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
