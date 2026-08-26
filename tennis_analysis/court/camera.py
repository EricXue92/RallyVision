"""从球场关键点标定完整相机（内参焦距 + 外参），支持 3D 投影。

标定策略：平面标定板（球场）单视角下无法稳定估计全部内参，
固定主点=图像中心、无畸变、fx=fy，只搜焦距：对数尺度扫描焦距，
每档 solvePnP 取重投影误差最小者，再用 cv2.calibrateCamera
(CALIB_USE_INTRINSIC_GUESS|CALIB_FIX_PRINCIPAL_POINT|CALIB_FIX_ASPECT_RATIO|
 CALIB_ZERO_TANGENT_DIST|CALIB_FIX_K1..K6) 以扫描出的最优焦距为初值细化
（离散网格扫描本身精度不够，细化步把焦距/位姿收敛到连续最优解）。
"""
import numpy as np
import cv2


class CameraModel:
    def __init__(self, K, rvec, tvec):
        self.K = np.asarray(K, dtype=float)
        self.rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
        self.tvec = np.asarray(tvec, dtype=float).reshape(3, 1)

    @classmethod
    def calibrate(cls, image_points, world_points_2d, image_size):
        img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        world = np.asarray(world_points_2d, dtype=np.float64)
        obj = np.column_stack([world, np.zeros(len(world))])
        w, h = image_size
        best = None
        for focal in np.geomspace(0.5 * w, 4.0 * w, 40):
            K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]])
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            cam = cls(K, rvec, tvec)
            err = cam.reprojection_error(img, world)
            if best is None or err < best[0]:
                best = (err, cam)
        if best is None:
            raise RuntimeError("相机标定失败 / camera calibration failed")

        # 细化：以扫描出的最优焦距为初值，用 calibrateCamera 联合优化焦距+位姿，
        # 主点/宽高比/畸变全部锁死，避免单视角平面标定板欠约束导致内参发散。
        init_K = best[1].K
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
        )
        try:
            _, K_ref, _, rvecs, tvecs = cv2.calibrateCamera(
                [obj.astype(np.float32)], [img.astype(np.float32)], (w, h),
                init_K.copy(), None, flags=flags,
            )
            refined = cls(K_ref, rvecs[0], tvecs[0])
            refined_err = refined.reprojection_error(img, world)
            if refined_err < best[0]:
                best = (refined_err, refined)
        except cv2.error:
            pass
        return best[1]

    def project(self, world_points):
        pts = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        proj, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, None)
        return proj.reshape(-1, 2)

    def reprojection_error(self, image_points, world_points_2d):
        return float(np.mean(self.per_point_errors(image_points, world_points_2d)))

    def per_point_errors(self, image_points, world_points_2d):
        """逐点重投影误差（px），np.ndarray[N]。"""
        img = np.asarray(image_points, dtype=float).reshape(-1, 2)
        world = np.asarray(world_points_2d, dtype=float)
        obj = np.column_stack([world, np.zeros(len(world))])
        return np.linalg.norm(self.project(obj) - img, axis=1)

    def to_dict(self):
        return {"K": self.K.tolist(), "rvec": self.rvec.ravel().tolist(), "tvec": self.tvec.ravel().tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(np.array(data["K"]), np.array(data["rvec"]), np.array(data["tvec"]))


def calibrate_with_outlier_rejection(
    image_points, world_points_2d, image_size, *, point_error_threshold_px=10.0, min_points=6
):
    """带外点剔除的标定（CourtCheck HomographyEstimator「按重投影误差择优」
    思想的移植，实现为迭代剔除而非 12 子集枚举——CameraModel 是全内外参标定，
    枚举 4 点子集欠约束，迭代剔最差点等价且适配任意点数）。

    某个关键点被检测模型稳定检错时（多帧中位数救不了），全点标定会被它
    系统性拉偏。这里：标定 → 逐点重投影误差 → 若最差点误差 > 阈值且剩余
    点数 > min_points，剔掉它重标定 → 循环。

    Returns:
        (CameraModel, inlier_mask)：inlier_mask 是与输入等长的 bool 数组，
        False = 被剔除的外点。至少保留 min_points 个点（保底后即使仍有
        超阈值点也停止剔除，由调用方的整体误差门限兜底降级）。
    """
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    world = np.asarray(world_points_2d, dtype=np.float64)
    inlier_mask = np.ones(len(img), dtype=bool)

    while True:
        camera = CameraModel.calibrate(img[inlier_mask], world[inlier_mask], image_size)
        if inlier_mask.sum() <= min_points:
            return camera, inlier_mask
        errors = camera.per_point_errors(img[inlier_mask], world[inlier_mask])
        worst_local = int(np.argmax(errors))
        if errors[worst_local] <= point_error_threshold_px:
            return camera, inlier_mask
        # errors 的索引是 inlier 子集内的局部索引，换算回全量索引再剔除
        worst_global = np.flatnonzero(inlier_mask)[worst_local]
        inlier_mask[worst_global] = False
