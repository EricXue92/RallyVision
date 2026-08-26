"""标定关键点剔除（CourtCheck 多子集单应性择优思想的移植）单测。

场景：14 个关键点里某个点被检测模型**稳定检错**（多帧中位数救不了），
全点一把梭标定会被它系统性拉偏 → 所有落点整体偏移。
calibrate_with_outlier_rejection 迭代剔除重投影误差最差的点后重标定，
返回 (CameraModel, inlier_mask)，保底 min_points 个点。

测试用合成相机：手工构造真值 K/R/t，把 COURT_KEYPOINTS_M 投影成像素点，
再人为污染其中一点，验证污染点被剔、其余保留、误差回到干净水平。
"""
import numpy as np
import cv2
import pytest

from tennis_analysis.court.camera import CameraModel, calibrate_with_outlier_rejection
from tennis_analysis.court.keypoint_detector import COURT_KEYPOINTS_M

IMAGE_SIZE = (1280, 720)


def _ground_truth_camera():
    """相机在球场近端底线后上方俯视球场中心（任意合法位姿即可，测试只关心
    投影一致性，不关心画面朝向美观）。"""
    w, h = IMAGE_SIZE
    focal = 1400.0
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]])

    center = np.array([5.485, 38.0, 9.0])          # 世界坐标（z 垂直于场地平面）
    target = np.array([5.485, 11.885, 0.0])        # 看向球场中心（网中点）
    z_cam = target - center
    z_cam = z_cam / np.linalg.norm(z_cam)
    x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.stack([x_cam, y_cam, z_cam], axis=0)
    tvec = (-R @ center).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return CameraModel(K, rvec, tvec)


def _synthetic_points():
    camera = _ground_truth_camera()
    world = COURT_KEYPOINTS_M
    obj = np.column_stack([world, np.zeros(len(world))])
    image_points = camera.project(obj)
    return image_points, world


def test_corrupted_point_is_rejected():
    image_points, world = _synthetic_points()
    corrupted = image_points.copy()
    corrupted[5] += np.array([32.0, -28.0])  # 单点稳定检错 ~40px

    camera, inlier_mask = calibrate_with_outlier_rejection(
        corrupted, world, IMAGE_SIZE, point_error_threshold_px=10.0, min_points=6
    )

    assert inlier_mask.dtype == bool and inlier_mask.shape == (14,)
    assert not inlier_mask[5], "被污染的点应被剔除"
    assert inlier_mask.sum() == 13, "其余点应全部保留"
    clean_error = camera.reprojection_error(image_points[inlier_mask], world[inlier_mask])
    assert clean_error < 2.0, f"剔除后应回到干净标定精度，实测 {clean_error:.2f}px"


def test_clean_points_keep_all_inliers():
    image_points, world = _synthetic_points()
    camera, inlier_mask = calibrate_with_outlier_rejection(
        image_points, world, IMAGE_SIZE, point_error_threshold_px=10.0, min_points=6
    )
    assert inlier_mask.all(), "无污染时不应剔除任何点"
    assert camera.reprojection_error(image_points, world) < 2.0


def test_rejection_stops_at_min_points_floor():
    image_points, world = _synthetic_points()
    corrupted = image_points.copy()
    rng = np.random.default_rng(7)
    for idx in range(9):  # 污染 9/14 个点，若无保底会剔到不足以标定
        corrupted[idx] += rng.uniform(25.0, 60.0, size=2)

    _, inlier_mask = calibrate_with_outlier_rejection(
        corrupted, world, IMAGE_SIZE, point_error_threshold_px=10.0, min_points=6
    )
    assert inlier_mask.sum() >= 6, "剔除必须保底 min_points 个点"


def test_system_calibration_rejects_stable_bad_keypoint(monkeypatch):
    """system._try_calibrate_from_median 接线验证（bare __new__ 手法，同
    test_system_orchestration.py）：一个被稳定检错 120px 的关键点不应把整个
    标定拉偏——返回的相机在其余 13 个干净点上的重投影误差要 < 2px。"""
    import tennis_analysis.system as system_module
    from tennis_analysis.system import TennisAnalysisSystem

    monkeypatch.setattr(system_module, "CameraModel", CameraModel, raising=False)
    monkeypatch.setattr(system_module, "COURT_KEYPOINTS_M", COURT_KEYPOINTS_M, raising=False)
    monkeypatch.setattr(
        system_module, "calibrate_with_outlier_rejection", calibrate_with_outlier_rejection, raising=False
    )

    image_points, world = _synthetic_points()
    corrupted = image_points.copy()
    corrupted[5] += np.array([85.0, -85.0])  # ~120px 稳定检错
    valid_mask = np.ones(14, dtype=bool)

    system = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    system.frame_width, system.frame_height = IMAGE_SIZE

    camera = system._try_calibrate_from_median(corrupted, valid_mask)

    assert camera is not None, "剔除外点后应标定成功而非降级"
    clean_mask = np.ones(14, dtype=bool)
    clean_mask[5] = False
    clean_error = camera.reprojection_error(image_points[clean_mask], world[clean_mask])
    assert clean_error < 2.0, f"污染点应被剔除，干净点误差实测 {clean_error:.2f}px"


def test_subset_input_smaller_than_14_supported():
    """system.py 传入的是 valid_mask 过滤后的子集（可能只有 8-10 个点），
    函数不能假定长度恒为 14。"""
    image_points, world = _synthetic_points()
    sub_img, sub_world = image_points[:9], world[:9]
    camera, inlier_mask = calibrate_with_outlier_rejection(
        sub_img, sub_world, IMAGE_SIZE, point_error_threshold_px=10.0, min_points=6
    )
    assert inlier_mask.shape == (9,)
    assert inlier_mask.all()
    assert camera.reprojection_error(sub_img, sub_world) < 2.0
