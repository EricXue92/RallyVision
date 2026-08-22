import numpy as np
from tennis_analysis.court.keypoint_detector import COURT_KEYPOINTS_M


def test_fourteen_points_geometry():
    kp = COURT_KEYPOINTS_M
    assert kp.shape == (14, 2)
    xs, ys = kp[:, 0], kp[:, 1]
    assert xs.min() == 0.0 and abs(xs.max() - 10.97) < 1e-9
    assert ys.min() == 0.0 and abs(ys.max() - 23.77) < 1e-9
    # 单打边线 x=1.37/9.60 各 4 点；发球线 y=5.485/18.285 各 3 点（含中点 T）
    assert (np.isclose(xs, 1.37).sum(), np.isclose(xs, 9.60).sum()) == (4, 4)
    assert np.isclose(ys, 5.485).sum() == 3 and np.isclose(ys, 18.285).sum() == 3
