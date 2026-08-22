import numpy as np
import cv2
from tennis_analysis.court.keypoint_detector import COURT_KEYPOINTS_M, _UPSTREAM_TO_CANONICAL_INDEX


def test_fourteen_points_geometry():
    kp = COURT_KEYPOINTS_M
    assert kp.shape == (14, 2)
    xs, ys = kp[:, 0], kp[:, 1]
    assert xs.min() == 0.0 and abs(xs.max() - 10.97) < 1e-9
    assert ys.min() == 0.0 and abs(ys.max() - 23.77) < 1e-9
    # 单打边线 x=1.37/9.60 各 4 点；发球线 y=5.485/18.285 各 3 点（含中点 T）
    assert (np.isclose(xs, 1.37).sum(), np.isclose(xs, 9.60).sum()) == (4, 4)
    assert np.isclose(ys, 5.485).sum() == 3 and np.isclose(ys, 18.285).sum() == 3


# 上游 yastrebksv/TennisCourtDetector court_reference.py::CourtReference 的像素参考坐标
# （Apache-2.0；未运行其仓库，逐字段抄自源码构造，与 keypoint_detector.py 模块 docstring
# 的推导表格一一对应）：
#   baseline_top      = ((286, 561), (1379, 561))
#   baseline_bottom    = ((286, 2935), (1379, 2935))
#   left_inner_line    = ((423, 561), (423, 2935))
#   right_inner_line   = ((1242, 561), (1242, 2935))
#   top_inner_line     = ((423, 1110), (1242, 1110))
#   bottom_inner_line  = ((423, 2386), (1242, 2386))
#   middle_line        = ((832, 1110), (832, 2386))
#   key_points = [*baseline_top, *baseline_bottom, *left_inner_line, *right_inner_line,
#                 *top_inner_line, *bottom_inner_line, *middle_line]
_UPSTREAM_REFERENCE_PX = np.array([
    [286, 561], [1379, 561], [286, 2935], [1379, 2935],
    [423, 561], [423, 2935], [1242, 561], [1242, 2935],
    [423, 1110], [1242, 1110], [423, 2386], [1242, 2386],
    [832, 1110], [832, 2386],
], dtype=np.float32)


def test_upstream_to_canonical_mapping_matches_reference_geometry():
    """回归钉住 `_UPSTREAM_TO_CANONICAL_INDEX`（本任务实际修复的 bug 就在这张表）。

    用上游 court_reference.py 的 4 个外角像素点与 COURT_KEYPOINTS_M 的 4 个外角
    世界坐标算单应性（这是纯几何参考图，非透视照片，理论上等价于一个仿射缩放），
    把全部 14 个上游参考点变换到米制世界坐标，逐点核对是否落在按当前映射表重排后
    的 COURT_KEYPOINTS_M 上。

    这条测试真正钉住了点序：如果把索引 5/6 的互换错误地"修复"回恒等映射
    （即本任务修复前的 bug 状态），变换后的点会落到相邻发球线/边线格的错误位置，
    偏差达到米级，必定超出 atol，测试会失败（已手动验证并记录在
    task-4-report.md 的 RED 证据里）。
    """
    src_corners = _UPSTREAM_REFERENCE_PX[[0, 1, 2, 3]]
    dst_corners = COURT_KEYPOINTS_M[[0, 1, 2, 3]].astype(np.float32)
    homography = cv2.getPerspectiveTransform(src_corners, dst_corners)

    warped = cv2.perspectiveTransform(
        _UPSTREAM_REFERENCE_PX.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)

    expected = COURT_KEYPOINTS_M[_UPSTREAM_TO_CANONICAL_INDEX]
    assert np.allclose(warped, expected, atol=0.05)
