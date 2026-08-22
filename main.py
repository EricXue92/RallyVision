import argparse
import os
from importlib.util import find_spec

from tennis_analysis.system import TennisAnalysisSystem, load_runtime_dependencies


def validate_cli_dependencies(args, parser):
    if args.player_detector == 'yolo-person' and args.person_tracker != 'none' and find_spec('lap') is None:
        parser.exit(
            2,
            "error: --person-tracker requires lap>=0.5.12.\n"
            "Install it with one of these commands:\n"
            "  pip install -r requirements.txt\n"
            "  pip install \"lap>=0.5.12\"\n"
            "Or disable MOT tracking with: --person-tracker none\n",
        )


def main():
    parser = argparse.ArgumentParser(description='网球比赛视频分析系统')
    parser.add_argument('--video-path', default='videos/demo.mp4', type=str, help='输入视频文件路径')
    parser.add_argument('--template-path', default='templates/demo.png', type=str, help='球场模板图像路径；不提供时会弹出文件选择框')
    parser.add_argument('--output-dir', default=None, type=str, help='输出目录，默认 outputs/<视频文件名>')
    parser.add_argument('--ball-model', default='weights/tennis-ball.pt', type=str, help='YOLO 网球检测模型路径')
    parser.add_argument('--ball-detector', default='yolo', choices=['yolo', 'tracknet', 'wasb'], type=str, help='球检测后端：yolo、tracknet 或 wasb')
    parser.add_argument('--tracknet-model', default='weights/tracknet_ball.pt', type=str, help='TrackNet 网球检测模型路径（--ball-detector tracknet 时使用）')
    parser.add_argument('--wasb-model', default='weights/wasb_tennis.pth', type=str, help='WASB-SBDT 网球检测模型路径（--ball-detector wasb 时使用）')
    parser.add_argument('--player-detector', default='yolo-person', choices=['pose', 'yolo-person'], help='球员检测方式：pose 使用姿态关键点，yolo-person 使用 YOLO 人框底部中点')
    parser.add_argument('--person-model', default='weights/yolo26s.pt', type=str, help='YOLO 人体目标检测模型路径或模型名，默认 weights/yolo26s.pt')
    parser.add_argument('--person-tracker', default='botsort', choices=['none', 'botsort', 'bytetrack'], help='YOLO 人体框多目标跟踪器：none、botsort 或 bytetrack，默认 botsort')
    parser.add_argument('--player-detect-interval', default=2, type=int, help='球员检测间隔帧数，1 表示每帧检测；实时预览可设为 2')
    parser.add_argument('--pose-family', default='rtmpose', choices=['rtmpose', 'rtmo', 'yolo-pose'], help='姿态模型族')
    parser.add_argument('--pose-mode', default='balanced', choices=['lightweight', 'balanced', 'performance'], help='RTMPose 模型档位')
    parser.add_argument('--yolo-pose-model', default='weights/yolo11s-pose.pt', type=str, help='YOLO pose 模型路径或模型名，默认 weights/yolo11s-pose.pt')
    parser.add_argument('--court-detection', default='auto-fallback', choices=['manual', 'auto', 'auto-fallback'], help='球场外角点检测方式：manual 手动点击，auto 自动检测，auto-fallback 自动失败后手动')
    parser.add_argument('--pose-roi', choices=['true', 'false'], default='true', help='是否显示姿态检测 ROI 框，默认 true')
    parser.add_argument('--display', choices=['true', 'false'], default='true', help='是否显示视频窗口，默认 true')
    parser.add_argument('--skeletons', choices=['true', 'false'], default='true', help='是否显示人体骨架，默认 true')
    parser.add_argument('--player-trajectories', choices=['true', 'false'], default='true', help='是否显示球员轨迹，默认 true')
    parser.add_argument('--court-trajectory', choices=['true', 'false'], default='true', help='是否显示球场轨迹，默认 true')
    parser.add_argument('--tennis-ball-trajectory', choices=['true', 'false'], default='true', help='是否显示网球轨迹，默认 true')
    parser.add_argument('--bounce-detection', choices=['true', 'false'], default='true', help='是否检测并标注网球弹跳点，默认 true')
    parser.add_argument('--bounce-classifier', default='', type=str, help='可选弹跳检测分类器 pkl 路径；不传时使用规则后处理')
    parser.add_argument('--mini-map', choices=['true', 'false'], default='true', help='是否显示球场小地图，默认 true')
    parser.add_argument('--court-calibration', default='keypoints', choices=['keypoints', 'homography'], help='相机标定方式：keypoints 用 14 点检测标定完整相机，homography 强制降级为仅单应性（无 speed/spin，只有 line_call），默认 keypoints')
    parser.add_argument('--keypoint-model', default='weights/court_keypoints.pt', type=str, help='球场关键点检测模型路径')
    parser.add_argument('--shot-metrics', choices=['true', 'false'], default='true', help='是否计算击球速度/旋转（segments/shot_metrics/spin），默认 true')
    parser.add_argument('--line-call', default='doubles', choices=['singles', 'doubles', 'off'], help='弹跳落点判罚：singles/doubles 场地模式，off 关闭，默认 doubles')
    parser.add_argument('--player-stats', choices=['true', 'false'], default='true', help='是否显示球员统计信息，默认 true')
    parser.add_argument('--save-images', action='store_true', default=False, help='保存处理后的图像')
    parser.add_argument('--performance-stats', action='store_true', default=False, help='显示性能统计信息')
    parser.add_argument('--visualize-positions', choices=['true', 'false'], default='true', help='是否生成球员位置热力图和散点图，默认 true')
    parser.add_argument('--audio', choices=['true', 'false'], default='true', help='是否保留原视频音频，默认 true')
    parser.add_argument('--language', default='en', choices=['zh', 'en'], help='选择界面语言 (zh/en)')
    args = parser.parse_args()
    validate_cli_dependencies(args, parser)

    load_runtime_dependencies()

    if args.language == 'en':
        from tennis_analysis.visualization.player_positions_en import analyze_player_positions
    else:
        from tennis_analysis.visualization.player_positions_zh import analyze_player_positions

    system = TennisAnalysisSystem(
        args.video_path,
        show_display=args.display == 'true',
        show_skeletons=args.skeletons == 'true',
        show_player_trajectories=args.player_trajectories == 'true',
        show_court_trajectory=args.court_trajectory == 'true',
        show_tennis_ball_trajectory=args.tennis_ball_trajectory == 'true',
        show_player_stats=args.player_stats == 'true',
        show_performance_stats=args.performance_stats,
        save_images=args.save_images,
        language=args.language,
        output_dir=args.output_dir,
        ball_model_path=args.ball_model,
        template_path=args.template_path,
        pose_mode=args.pose_mode,
        pose_family=args.pose_family,
        yolo_pose_model=args.yolo_pose_model,
        player_detector=args.player_detector,
        person_model=args.person_model,
        person_tracker=args.person_tracker,
        player_detect_interval=args.player_detect_interval,
        show_pose_roi=args.pose_roi == 'true',
        court_detection=args.court_detection,
        show_bounce_detection=args.bounce_detection == 'true',
        bounce_classifier_path=args.bounce_classifier,
        show_mini_map=args.mini_map == 'true',
        ball_detector=args.ball_detector,
        tracknet_model_path=args.tracknet_model,
        wasb_model_path=args.wasb_model,
        court_calibration=args.court_calibration,
        keypoint_model_path=args.keypoint_model,
        shot_metrics=args.shot_metrics == 'true',
        line_call=args.line_call,
    )

    system.keep_audio = args.audio == 'true'
    system.process_video()

    if args.visualize_positions == 'true':
        print("\n开始生成球员位置可视化...")
        analyze_player_positions(system.detections_path, os.path.join(system.save_dir, 'position_visualizations'), fps=system.fps)
        print("球员位置可视化完成")

if __name__ == "__main__":
    main()

