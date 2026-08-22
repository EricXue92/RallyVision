"""固定机位相机标定的纯逻辑：逐关键点中位数聚合 + 漂移判定（Task 10 amended brief）。

拆成纯函数（只用 numpy，不碰 cv2/torch）方便脱离视频单测——真正调用关键点检测模型 /
CameraModel.calibrate 的编排逻辑在 system.py::TennisAnalysisSystem._calibrate_camera：
对视频前 300 帧每 10 帧抽 1 帧检测 -> median_keypoints_over_frames 逐点中位数
-> CameraModel.calibrate 整段复用；此后每 900 帧重检 1 帧，keypoints_drifted
判定相机是否被碰，命中则用其后 300 帧重新标定（同一套采样 -> 中位数流程）。
"""
import numpy as np


def median_keypoints_over_frames(keypoint_frames, min_valid_frames=3):
    """对多帧关键点检测结果逐点取中位数。

    Args:
        keypoint_frames: list[np.ndarray[14,2]]，每帧的关键点像素坐标（低置信点为 NaN）。
            至少要有 1 帧。
        min_valid_frames: 某关键点在采样帧中有效（非 NaN）次数 < 此值时，整体判定为无效
            ——即使个别帧检测到了，也认为不够稳定，中位数记 NaN、valid_mask 记 False。

    Returns:
        (median_points, valid_mask)：median_points 是 np.ndarray[14,2]（无效点为 NaN），
        valid_mask 是 np.ndarray[14] bool。
    """
    stacked = np.stack([np.asarray(frame, dtype=float) for frame in keypoint_frames], axis=0)
    valid_counts = np.sum(~np.isnan(stacked).any(axis=2), axis=0)  # [14]
    with np.errstate(invalid="ignore"):
        median_points = np.nanmedian(stacked, axis=0)
    valid_mask = valid_counts >= min_valid_frames
    median_points = np.where(valid_mask[:, None], median_points, np.nan)
    return median_points, valid_mask


def keypoints_drifted(current_points, baseline_median, baseline_valid_mask, threshold_px=10.0, min_keypoints=4):
    """判定单帧关键点检测相对标定基线是否达到「相机被碰」的漂移量级。

    只比较标定基线有效、且当前帧也检测到（非 NaN）的关键点；两者欧氏距离 > threshold_px
    的点数 >= min_keypoints 时判定为漂移。基线无效或当前帧未检测到的点不参与比较（既不能
    算漂移也不能算未漂移——没有可信数据）。

    Args:
        current_points: np.ndarray[14,2]，漂移守卫重检帧的关键点检测结果（低置信点为 NaN）。
        baseline_median: np.ndarray[14,2]，标定时的逐点中位数（median_keypoints_over_frames 输出）。
        baseline_valid_mask: np.ndarray[14] bool，标定时的逐点有效掩码（同上）。

    Returns:
        bool
    """
    current = np.asarray(current_points, dtype=float)
    current_valid = ~np.isnan(current).any(axis=1)
    comparable = current_valid & np.asarray(baseline_valid_mask, dtype=bool)
    if not np.any(comparable):
        return False
    deltas = np.linalg.norm(current[comparable] - np.asarray(baseline_median, dtype=float)[comparable], axis=1)
    drifted_count = int(np.sum(deltas > threshold_px))
    return drifted_count >= min_keypoints
