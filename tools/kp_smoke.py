"""球场 14 关键点检测冒烟测试脚本 / Court keypoint detector smoke test.

对视频第一帧跑 CourtKeypointDetector.detect，把有效点画上编号存
outputs/kp_smoke.png，并用检测点 + 映射后的世界坐标喂给
CameraModel.calibrate，打印重投影误差，作为「点序映射对不对」的
端到端核验（Task 4 brief Step 4，见 .superpowers/sdd/2026-08-22-
rallyvision-phase2-plan/task-4-report.md）。

Usage:
    uv run python tools/kp_smoke.py
    uv run python tools/kp_smoke.py --video-path videos/demo1.mp4
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DEFAULT_VIDEO_PATH = os.path.join(REPO_ROOT, "videos", "demo.mp4")
DEFAULT_WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "court_keypoints.pt")
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "outputs", "kp_smoke.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", default=DEFAULT_VIDEO_PATH, help="输入视频路径 / input video path")
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="权重路径 / weights path")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="标注图输出路径 / annotated output image path")
    args = parser.parse_args()

    if not os.path.isfile(args.weights_path):
        print(f"缺权重 / weights missing: {args.weights_path}")
        print("见 weights/README.md 的下载说明 / see weights/README.md for download instructions")
        sys.exit(0)

    import cv2
    import numpy as np
    from tennis_analysis.court.keypoint_detector import CourtKeypointDetector, COURT_KEYPOINTS_M
    from tennis_analysis.court.camera import CameraModel

    cap = cv2.VideoCapture(args.video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"无法读取视频第一帧 / failed to read first frame: {args.video_path}")
        sys.exit(1)
    print(f"帧尺寸 / frame size: {frame.shape}")

    detector = CourtKeypointDetector(args.weights_path)
    points = detector.detect(frame)
    if points is None:
        print("检测失败，有效点不足 6 个 / detection failed, <6 valid points")
        sys.exit(1)

    valid_mask = ~np.isnan(points).any(axis=1)
    valid_count = int(valid_mask.sum())
    print(f"有效点数 / valid points: {valid_count}/14")

    vis = frame.copy()
    for i, (p, v) in enumerate(zip(points, valid_mask)):
        if not v:
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, vis)
    print(f"已保存标注图 / saved annotated image: {args.output_path}")

    # 端到端核验点序：把检测到的有效点 + 对应世界坐标喂给 CameraModel.calibrate，
    # 重投影误差小说明点序映射（COURT_KEYPOINTS_M 顺序）是对的。
    img_pts = points[valid_mask]
    world_pts = COURT_KEYPOINTS_M[valid_mask]
    if len(img_pts) < 6:
        print("有效点不足 6 个，无法标定 / <6 valid points, cannot calibrate")
        sys.exit(1)

    h, w = frame.shape[:2]
    cam = CameraModel.calibrate(img_pts, world_pts, (w, h))
    err = cam.reprojection_error(img_pts, world_pts)
    print(f"用于标定的点数 / points used for calibration: {len(img_pts)}")
    print(f"重投影误差 / reprojection error: {err:.3f} px")


if __name__ == "__main__":
    main()
