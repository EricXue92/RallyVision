import numpy as np
import cv2
from tennis_analysis.court.camera import CameraModel
from tennis_analysis.court.keypoint_detector import COURT_KEYPOINTS_M

W, H = 1280, 720


def _lookat_extrinsics(cam_pos, target):
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])          # world->cam
    tvec = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


# 标准场 14 点 (x,y)，米。规范定义在 tennis_analysis/court/keypoint_detector.py
# 的 COURT_KEYPOINTS_M，这里直接复用同一份数据，避免重复字面量（Task 4）。
COURT_POINTS = COURT_KEYPOINTS_M


def test_calibrate_recovers_projection():
    K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
    rvec, tvec = _lookat_extrinsics(np.array([5.485, -6.0, 3.0]), np.array([5.485, 11.885, 1.0]))
    truth = CameraModel(K, rvec, tvec)
    obs = truth.project(np.column_stack([COURT_POINTS, np.zeros(len(COURT_POINTS))]))
    cam = CameraModel.calibrate(obs, COURT_POINTS, (W, H))
    assert cam.reprojection_error(obs, COURT_POINTS) < 2.0
    # 空中点投影一致（标定出的相机对非地面点也要接近真值）
    air = np.array([[5.485, 11.885, 2.5]])
    assert np.linalg.norm(cam.project(air)[0] - truth.project(air)[0]) < 8.0


def test_serialization_roundtrip():
    K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
    rvec, tvec = _lookat_extrinsics(np.array([5.485, -6.0, 3.0]), np.array([5.485, 11.885, 1.0]))
    cam = CameraModel(K, rvec, tvec)
    cam2 = CameraModel.from_dict(cam.to_dict())
    pt = np.array([[3.0, 20.0, 1.5]])
    assert np.allclose(cam.project(pt), cam2.project(pt), atol=1e-6)
