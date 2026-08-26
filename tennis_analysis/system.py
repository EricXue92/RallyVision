import json
import os
import tempfile
from tkinter import filedialog
import tkinter as tk
import time
import argparse

# Task 7：比赛层（shot_type/rally/point_outcome/scoring/stats/highlights）纯 stdlib
# + numpy 编排，不依赖 cv2/torch，直接顶层 import 即可（不必走下面 load_runtime_dependencies
# 的懒加载——那套是为了让 --help 在没装重依赖时也能跑，这几个模块不属于重依赖）。
from .analysis.match_layer import run_match_layer
from .export.highlights import export_highlights, select_highlight_rallies


def load_runtime_dependencies():
    """Load heavy runtime dependencies after argparse has handled --help."""
    global cv2, np, YOLO, CourtMapper, annotate_court, compute_expanded_roi, PlayerTracker
    global CourtLineAutoDetector, CourtTrajectoryVisualizer, TennisBallTracker
    global BounceDetector, MiniMapVisualizer
    global PlayerPoseVisualizer, StatsVisualizer, RTMPoseProcessor, YOLOPoseProcessor, YOLOPersonDetector, vap
    global JsonlDetectionWriter, write_json, SCHEMA_VERSION
    global CourtKeypointDetector, COURT_KEYPOINTS_M, CameraModel, calibrate_with_outlier_rejection
    global median_keypoints_over_frames, keypoints_drifted
    global compute_shot_metrics_entries, write_shot_metrics, call_bounce
    global TrackNetBallDetector, TrackNetBallTrackerAdapter, compute_far_roi_rect
    global WASBBallDetector, WASBBallTrackerAdapter

    yolo_config_dir = os.path.join(tempfile.gettempdir(), "good-tennis-ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from .court.mapper import CourtMapper as _CourtMapper, annotate_court as _annotate_court
        from .court.mapper import compute_expanded_roi as _compute_expanded_roi
        from .court.auto_detector import CourtLineAutoDetector as _CourtLineAutoDetector
        from .tracking.player import PlayerTracker as _PlayerTracker
        from .analysis.bounce import BounceDetector as _BounceDetector
        from .visualization.court_trajectory import CourtTrajectoryVisualizer as _CourtTrajectoryVisualizer
        from .visualization.minimap import MiniMapVisualizer as _MiniMapVisualizer
        from .detection.tennis_ball import TennisBallTracker as _TennisBallTracker
        from .visualization.player_pose import PlayerPoseVisualizer as _PlayerPoseVisualizer
        from .visualization.stats import StatsVisualizer as _StatsVisualizer
        from .detection.rtmpose import RTMPoseProcessor as _RTMPoseProcessor
        from .detection.yolo_pose import YOLOPoseProcessor as _YOLOPoseProcessor
        from .detection.yolo_person import YOLOPersonDetector as _YOLOPersonDetector
        from .media import video_audio as _vap
        from .data.writer import JsonlDetectionWriter as _JsonlDetectionWriter
        from .data.writer import write_json as _write_json
        from .data.writer import SCHEMA_VERSION as _SCHEMA_VERSION
        from .court.keypoint_detector import CourtKeypointDetector as _CourtKeypointDetector
        from .court.keypoint_detector import COURT_KEYPOINTS_M as _COURT_KEYPOINTS_M
        from .court.camera import CameraModel as _CameraModel
        from .court.camera import calibrate_with_outlier_rejection as _calibrate_with_outlier_rejection
        from .court.camera_calibration import median_keypoints_over_frames as _median_keypoints_over_frames
        from .court.camera_calibration import keypoints_drifted as _keypoints_drifted
        from .analysis.shot_pipeline import compute_shot_metrics_entries as _compute_shot_metrics_entries
        from .analysis.shot_pipeline import write_shot_metrics as _write_shot_metrics
        from .analysis.line_call import call_bounce as _call_bounce
        from .detection.tracknet_ball import TrackNetBallDetector as _TrackNetBallDetector
        from .detection.tracknet_ball import TrackNetBallTrackerAdapter as _TrackNetBallTrackerAdapter
        from .detection.tracknet_ball import compute_far_roi_rect as _compute_far_roi_rect
        from .detection.wasb_ball import WASBBallDetector as _WASBBallDetector
        from .detection.wasb_ball import WASBBallTrackerAdapter as _WASBBallTrackerAdapter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.backends.cudnn.benchmark = True
            _torch.backends.cuda.matmul.allow_tf32 = True
            _torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    CourtMapper = _CourtMapper
    annotate_court = _annotate_court
    compute_expanded_roi = _compute_expanded_roi
    CourtLineAutoDetector = _CourtLineAutoDetector
    PlayerTracker = _PlayerTracker
    BounceDetector = _BounceDetector
    CourtTrajectoryVisualizer = _CourtTrajectoryVisualizer
    MiniMapVisualizer = _MiniMapVisualizer
    TennisBallTracker = _TennisBallTracker
    PlayerPoseVisualizer = _PlayerPoseVisualizer
    StatsVisualizer = _StatsVisualizer
    RTMPoseProcessor = _RTMPoseProcessor
    YOLOPoseProcessor = _YOLOPoseProcessor
    YOLOPersonDetector = _YOLOPersonDetector
    vap = _vap
    JsonlDetectionWriter = _JsonlDetectionWriter
    write_json = _write_json
    SCHEMA_VERSION = _SCHEMA_VERSION
    CourtKeypointDetector = _CourtKeypointDetector
    COURT_KEYPOINTS_M = _COURT_KEYPOINTS_M
    CameraModel = _CameraModel
    calibrate_with_outlier_rejection = _calibrate_with_outlier_rejection
    median_keypoints_over_frames = _median_keypoints_over_frames
    keypoints_drifted = _keypoints_drifted
    compute_shot_metrics_entries = _compute_shot_metrics_entries
    write_shot_metrics = _write_shot_metrics
    call_bounce = _call_bounce
    TrackNetBallDetector = _TrackNetBallDetector
    TrackNetBallTrackerAdapter = _TrackNetBallTrackerAdapter
    compute_far_roi_rect = _compute_far_roi_rect
    WASBBallDetector = _WASBBallDetector
    WASBBallTrackerAdapter = _WASBBallTrackerAdapter

class TennisAnalysisSystem:
    def __init__(self, video_path, show_display=True, 
                 show_skeletons=True, show_player_trajectories=True, 
                 show_court_trajectory=True, show_tennis_ball_trajectory=True,
                 show_player_stats=True, show_performance_stats=False, 
                 save_images=False, language='zh', output_dir=None,
                 ball_model_path='weights/tennis-ball.pt', template_path=None,
                 pose_mode='balanced', pose_family='rtmpose',
                 yolo_pose_model='weights/yolo11s-pose.pt', player_detector='pose',
                 person_model='weights/yolo26s.pt', person_tracker='none',
                 player_detect_interval=1,
                 show_pose_roi=True,
                 court_detection='auto-fallback', show_bounce_detection=True,
                 bounce_classifier_path='', show_mini_map=True,
                 court_match_width=320,
                 ball_detector='yolo', tracknet_model_path='weights/tracknet_ball.pt',
                 wasb_model_path='weights/wasb_tennis.pth',
                 court_calibration='keypoints', keypoint_model_path='weights/court_keypoints.pt',
                 shot_metrics=True, line_call='doubles',
                 match_scoring=False, first_server='lower',
                 upper_hand='right', lower_hand='right',
                 best_of=3, no_ad=False, highlights=False, far_roi=True):
        self.video_path = video_path
        self.show_display = show_display
        self.language = language
        self.template_path = template_path
        self.ball_model_path = ball_model_path
        self.pose_mode = pose_mode
        self.pose_family = pose_family
        self.yolo_pose_model = yolo_pose_model
        self.player_detector = player_detector
        self.person_model = person_model
        self.person_tracker = person_tracker
        self.player_detect_interval = max(1, int(player_detect_interval))
        self.show_pose_roi = show_pose_roi
        self.court_detection = court_detection
        # 未显式指定弹跳分类器时,默认启用 CatBoost 落点模型(缺权重则回退规则评分)
        if not bounce_classifier_path and os.path.exists('weights/ctb_regr_bounce.cbm'):
            bounce_classifier_path = 'weights/ctb_regr_bounce.cbm'
        self.bounce_classifier_path = bounce_classifier_path
        self.court_match_width = court_match_width

        # Task 10: shot metrics / spin / line calling 主流程接线
        self.ball_detector = ball_detector  # 'yolo' | 'tracknet' | 'wasb'
        self.far_roi = bool(far_roi)  # 远场 ROI 二次推理开关（仅 tracknet 后端生效）
        self._far_roi_rect_cache = None
        self._far_roi_rect_computed = False
        self.tracknet_model_path = tracknet_model_path
        self.wasb_model_path = wasb_model_path
        self.court_calibration = court_calibration  # 'keypoints' | 'homography'（强制降级）
        self.keypoint_model_path = keypoint_model_path
        self.enable_shot_metrics = bool(shot_metrics)
        self.line_call_mode = None if line_call in (None, 'off') else line_call  # None = 跳过 call_bounce
        self.camera_dict = None

        # Task 7：比赛层（shot_type/rally/point_outcome/scoring/stats/highlights）主流程接线
        self.enable_match_scoring = bool(match_scoring)
        self.first_server = first_server            # 'upper' | 'lower'
        self.upper_hand = upper_hand                # 'right' | 'left'
        self.lower_hand = lower_hand                # 'right' | 'left'
        self.best_of = int(best_of)
        self.sets_to_win = 3 if self.best_of == 5 else 2
        self.no_ad = bool(no_ad)
        self.enable_highlights = bool(highlights)


        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_court_trajectory = show_court_trajectory
        self.show_tennis_ball_trajectory = show_tennis_ball_trajectory
        self.show_bounce_detection = show_bounce_detection
        self.show_mini_map = show_mini_map
        self.show_player_stats = show_player_stats
        self.show_performance_stats = show_performance_stats
        self.save_images = save_images  

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Input video not found: {self.video_path}\n"
                "Pass a valid video file with --video-path."
            )
        if self.ball_detector == 'wasb' and not os.path.exists(self.wasb_model_path):
            # WASB 是可选备用后端（Task 9b delta）：缺权重时不整体中断运行，
            # 打印清晰双语提示后降级回默认的 yolo 后端（与下面 yolo 分支的
            # 权重存在性检查共用同一套校验，降级后走同一条路径）。
            print(
                f"[WASBBallDetector] 缺权重，降级为 yolo 球检测后端 / "
                f"weights missing, falling back to the yolo ball-detection backend: "
                f"{self.wasb_model_path}\n"
                "下载方式见 weights/README.md「WASB ball detector」章节 / "
                "See weights/README.md, the \"WASB ball detector\" section, for the download steps."
            )
            self.ball_detector = 'yolo'
        if self.ball_detector == 'yolo' and not os.path.exists(self.ball_model_path):
            raise FileNotFoundError(
                f"Ball detection model not found: {self.ball_model_path}\n"
                "Download or train a YOLO tennis ball model and place it at "
                "weights/tennis-ball.pt, or pass its path with --ball-model."
            )
        if self.ball_detector == 'tracknet' and not os.path.exists(self.tracknet_model_path):
            raise FileNotFoundError(
                f"TrackNet ball detection model not found: {self.tracknet_model_path}\n"
                "Download the TrackNet weights and place them at weights/tracknet_ball.pt, "
                "or pass --tracknet-model."
            )

        self.person_detector = None
        if self.player_detector == 'yolo-person':
            self.rtmpose_processor = None
            self.person_detector = YOLOPersonDetector(
                model_path=self.person_model,
                tracker=self.person_tracker,
            )
        elif self.pose_family == 'yolo-pose':
            self.rtmpose_processor = YOLOPoseProcessor(model_path=self.yolo_pose_model)
        else:
            self.rtmpose_processor = RTMPoseProcessor(mode=self.pose_mode, pose_family=self.pose_family)

        if self.ball_detector in ('tracknet', 'wasb'):
            self.yolo_ball_model = None
        else:
            self.yolo_ball_model = YOLO(self.ball_model_path)

        self.last_stats_update_frame = 0


        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)[:-4]
        self.save_dir = output_dir or os.path.join('outputs', self.video_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_save_dir = os.path.join(self.save_dir, 'detect_images')
        os.makedirs(self.images_save_dir, exist_ok=True)
        

        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.detections_path = os.path.join(self.save_dir, "detections.jsonl")
        self.bounce_events_path = os.path.join(self.save_dir, "bounce_events.json")
        self.cleaned_ball_trajectory_path = os.path.join(self.save_dir, "cleaned_ball_trajectory.json")
        self.shot_metrics_path = os.path.join(self.save_dir, "shot_metrics.json")
        self.match_score_path = os.path.join(self.save_dir, "match_score.json")
        self.match_stats_path = os.path.join(self.save_dir, "match_stats.json")
        self.highlights_path = os.path.join(self.save_dir, "highlights.mp4")
        self.output_video_path = os.path.join(self.save_dir, f"detect_{self.video_name}.mp4")
        self.temp_output_video_path = None
        self.detection_writer = None
        self.cached_player_detection = None
        

        self.player_1_hand = "right"  
        self.player_2_hand = "right"  
        self.start_time = None
        self.end_time = None
        

        if self.ball_detector == 'tracknet':
            tracknet_detector = TrackNetBallDetector(model_path=self.tracknet_model_path)
            self.tennis_ball_tracker = TrackNetBallTrackerAdapter(tracknet_detector, trajectory_length=30)
        elif self.ball_detector == 'wasb':
            wasb_detector = WASBBallDetector(model_path=self.wasb_model_path)
            self.tennis_ball_tracker = WASBBallTrackerAdapter(wasb_detector, trajectory_length=30)
        else:
            self.tennis_ball_tracker = TennisBallTracker(
                yolo_ball_model=self.yolo_ball_model,
                trajectory_length=30,
                show_trajectory=False,
                show_performance_stats=self.show_performance_stats
            )
        
        self.player_pose_visualizer = PlayerPoseVisualizer(
            rtmpose_processor=self.rtmpose_processor,
            person_detector=self.person_detector,
            player_detector=self.player_detector,
            show_skeletons=self.show_skeletons,
            show_player_trajectories=self.show_player_trajectories,
            show_performance_stats=self.show_performance_stats
        )
        

        self.court_trajectory_visualizer = CourtTrajectoryVisualizer()
        self.minimap_visualizer = MiniMapVisualizer()
        

        self.stats_update_interval_frames = 0
        self.cached_movement_stats = {}

        self.is_court_view_count = 0
        self.consecutive_non_court_frames = 0
        self.rally_active = False
        self.rally_count = 0  
        self.fps = 30  
        self.court_view_frames_threshold = 5
        self.non_court_frames_threshold = 5

        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0
        self.bounce_detector = None
        self.court_detection_result = None
    def process_video(self):
        """Process the input video."""
        self.start_time = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        video_duration = total_frames / fps
        

        self.fps = fps
        self.total_frames = total_frames


        template_frame = self._select_template_frame_from_video()
        if template_frame is not None:
            template_path = "<video-frame>"   # metadata 溯源标记,不是文件路径
            template_gray, template_color = self._template_from_frame(template_frame)
        else:
            template_path = self._get_template_path()
            template_gray, template_color = self._load_template(template_path, cap)
        

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = self._setup_video_writer(self.frame_width, self.frame_height, fps)


        corners, roi_corners, mid_height = self._setup_court_annotation(template_color)
        self.court_corners = corners
        self.court_roi_corners = roi_corners

        self._write_metadata(fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height)
        self.detection_writer = JsonlDetectionWriter(self.detections_path)
        

        self.court_mapper = CourtMapper(corners)
        self.player_pose_visualizer.court_mapper = self.court_mapper
        self.player_tracker = PlayerTracker(corners=corners, threshold=mid_height, history_size=30,
                                          detection_writer=self.detection_writer, fps=fps)
        self.bounce_detector = BounceDetector(fps=fps, classifier_path=self.bounce_classifier_path)
        

        self.stats_visualizer = StatsVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            language=self.language
        )
        
        frame_count = 0
        detect_frame_count = 0


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame, detect_frame_count = self._process_frame(frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)

        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        print(f"\n处理完成:")
        print(f"原始视频时长: {video_duration:.2f} 秒")
        print(f"处理耗时: {processing_time:.2f} 秒")
        print(f"处理速度比: {processing_time/video_duration:.2f}x")

        if detect_frame_count == 0:
            # 全程没有一帧通过 is_court_view 模板匹配:没有任何可分析数据。跳过弹跳
            # 后处理/比赛层/视频合成(它们会在空 temp 视频上二次崩溃),只释放资源后
            # 以退出码 0 结束——match_score/match_stats 不落盘,worker 的 report_builder
            # 会据此上报 court_not_detected;若在这里非零退出,worker 只能归为
            # pipeline_error,用户端就丢失了「未识别到球场画面」这个具体原因。
            print(
                f"未识别到球场画面: 0/{frame_count} 帧通过模板匹配,跳过后处理 / "
                f"no court-view frames detected ({frame_count} frames scanned), skipping post-processing"
            )
            self._release_capture_resources(cap)
            return

        self._cleanup(cap)

    def _active_ball_model_path(self):
        """当前生效的球检测模型路径（三选一，供 metadata 记录）。"""
        if self.ball_detector == 'tracknet':
            return self.tracknet_model_path
        if self.ball_detector == 'wasb':
            return self.wasb_model_path
        return self.ball_model_path

    def _write_metadata(self, fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height):
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": self.video_path,
                "name": self.video_name,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_sec": float(video_duration),
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "models": {
                "tennis_ball": self._active_ball_model_path(),
                "ball_detector": self.ball_detector,
                "player_detector": self.player_detector,
                "person": self.person_model if self.player_detector == 'yolo-person' else None,
                "person_tracker": self.person_tracker if self.player_detector == 'yolo-person' else None,
                "pose_family": self.pose_family if self.player_detector == 'pose' else None,
                "player_detect_interval": self.player_detect_interval,
            },
            "analysis": {
                "court_detection": self.court_detection,
                "bounce_detection": self.show_bounce_detection,
                "bounce_method": (
                    "rule_lag20_postprocess" if not self.bounce_classifier_path
                    else "catboost_lag2" if self.bounce_classifier_path.endswith(".cbm")
                    else "clf_lag20_postprocess"
                ),
                "bounce_classifier": self.bounce_classifier_path,
                "mini_map": self.show_mini_map,
                "court_calibration": self.court_calibration,
                "shot_metrics": self.enable_shot_metrics,
                "line_call": self.line_call_mode,
            },
            "court": {
                "template_path": template_path,
                "corners": corners,
                "roi_corners": roi_corners,
                "mid_height": mid_height,
                "detection_result": self.court_detection_result,
                "coordinate_system": {
                    "unit": "meter",
                    "width": 10.97,
                    "length": 23.77,
                },
            },
            "outputs": {
                "video": self.output_video_path,
                "detections": self.detections_path,
                "bounce_events": self.bounce_events_path,
                "cleaned_ball_trajectory": self.cleaned_ball_trajectory_path,
                "shot_metrics": self.shot_metrics_path if self.enable_shot_metrics else None,
            },
            # 相机标定结果要到 _cleanup 阶段（回合切分完才能标定）才有；这里先占位 null，
            # _patch_metadata_with_camera 会在标定/降级判定完成后回填（Task 10）。
            "camera": None,
        }
        write_json(self.metadata_path, metadata)

    def _process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):

        gray_frame = self._prepare_court_match_frame(frame, template_gray)
        
        # frame = self.draw_court_roi(frame, corners, roi_corners)

        is_court = self.is_court_view(gray_frame, template_gray)
        
        if is_court:
            self.is_court_view_count += 1
            self.consecutive_non_court_frames = 0
        else:
            self.consecutive_non_court_frames += 1
            self.is_court_view_count = 0
            

        if self.is_court_view_count >= self.court_view_frames_threshold and not self.rally_active:
            self.rally_active = True

            self.rally_count += 1

            self.player_tracker.start_new_rally()
            

        if self.consecutive_non_court_frames >= self.non_court_frames_threshold and self.rally_active:
            self.rally_active = False

            self.tennis_ball_tracker.clear_trajectory()
            self.cached_player_detection = None
            if self.bounce_detector is not None:
                self.bounce_detector.clear()


        if not is_court:
            return frame, detect_frame_count

        detect_frame_count += 1

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        roi = frame[y1:y2, x1:x2]
        if self.show_pose_roi:
            cv2.rectangle(frame, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
            cv2.putText(frame, "Pose ROI", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


        centroids, point_left_hands, point_right_hands = self._detect_players(roi, x1, y1, detect_frame_count)

        if self.ball_detector in ('tracknet', 'wasb'):
            # Task 9b Step 1：球员框 gating 只接入 TrackNet/WASB 两个热图后端
            # （brief 明确范围），YOLO 后端沿用旧调用不传 player_centers。
            player_centers = self._player_centers_for_gating(centroids)
            if self.ball_detector == 'tracknet':
                # 远场 ROI 二次推理仅接入 tracknet 后端（WASB 是备用后端，签名不动）
                detected_ball_position = self.tennis_ball_tracker.detect_ball(
                    frame, roi_corners=roi_corners, player_centers=player_centers,
                    far_roi_rect=self._far_roi_rect_for_ball(),
                )
            else:
                detected_ball_position = self.tennis_ball_tracker.detect_ball(
                    frame, roi_corners=roi_corners, player_centers=player_centers
                )
        else:
            detected_ball_position = self.tennis_ball_tracker.detect_ball(frame, roi_corners=roi_corners)
        ball_position = self.tennis_ball_tracker.update_trajectory(detected_ball_position, roi_corners)
        ball_court_position = self.court_mapper.image_to_court(ball_position) if ball_position != [0, 0] else None
        bounce_event = None
        

        players = self.player_tracker.update(frame_count, centroids, ball_position, 
                                             point_left_hands, point_right_hands, detect_frame_count,
                                             ball_court_position=ball_court_position,
                                             bounce_event=bounce_event)
        

        if frame_count == 1 or not self.cached_movement_stats:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.stats_update_interval_frames = int(self.player_tracker.fps * 0.5)

        if frame_count - self.last_stats_update_frame >= self.stats_update_interval_frames:

            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.last_stats_update_frame = frame_count


        t0 = time.time()

        # rally_count 不再在第一遍画:镜头切换计数对固定机位恒为 1(误导),
        # 真回合数由 _finalize_bounce_detection 的第二遍标注按 extract_rallies 切分补画。
        self.player_pose_visualizer.draw_players(
            frame=frame,
            player_tracker=self.player_tracker,
            cached_movement_stats=self.cached_movement_stats,
            stats_visualizer=self.stats_visualizer if self.show_player_stats else None,
        )
        t1 = time.time()
        if self.show_performance_stats:
            print(f"Drawing players took {t1 - t0:.2f} sec")
        

        if self.show_court_trajectory and not self.show_mini_map:
            t0 = time.time()
            frame = self.court_trajectory_visualizer.draw_overlay(frame, self.player_tracker.court_history)
            t1 = time.time()
            if self.show_performance_stats:
                print(f"Drawing court trajectory took {t1 - t0:.2f} sec")

        if self.show_mini_map:
            frame = self.minimap_visualizer.draw(
                frame,
                self.player_tracker.court_history,
                ball_court_position=None,
                bounce_events=[],
            )
        

        if frame is not None:
            if self.show_display:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            out.write(frame)

            if self.save_images:
                cv2.imwrite(os.path.join(self.images_save_dir, f"{frame_count}.png"), frame)
        return frame, detect_frame_count

    def _should_run_detection(self, detect_frame_count, interval):
        return (detect_frame_count - 1) % interval == 0

    def _detect_players(self, roi, x1, y1, detect_frame_count):
        if self._should_run_detection(detect_frame_count, self.player_detect_interval) or self.cached_player_detection is None:
            self.cached_player_detection = self.player_pose_visualizer.detect_players(roi, x1, y1)
        return self.cached_player_detection

    @staticmethod
    def _player_centers_for_gating(centroids):
        """把 `_detect_players` 返回的 centroids 统一抽取成 (x,y) 点列表，供
        Task 9b 球员框 gating（`tracknet_ball.is_implausible_ball_jump`）使用。

        centroids 元素格式因 `player_detector` 模式而异（见
        visualization/player_pose.py）：'yolo-person' 模式下是
        `{'point': (x,y), 'track_id': ...}` dict，'pose' 模式下是裸 (x,y)
        tuple。这里统一成点列表，坐标已是全帧空间（ROI 偏移在
        detect_players 内部加过），与球检测坐标系一致。"""
        points = []
        for centroid in centroids:
            point = centroid.get("point") if isinstance(centroid, dict) else centroid
            if point is not None:
                points.append(point)
        return points

    def _get_template_path(self):
        """Get the court template image path."""
        if self.template_path:
            if not os.path.exists(self.template_path):
                raise FileNotFoundError(
                    f"Court template image not found: {self.template_path}"
                )
            return self.template_path

        try:
            root = tk.Tk()
            root.withdraw()
            template_path = filedialog.askopenfilename(
                title="Select court template image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Unable to open the template picker. In headless environments, "
                "pass a court template image path with --template-path."
            ) from exc

        if not template_path:
            raise RuntimeError(
                "No court template image selected. Pass --template-path to run "
                "without the file picker."
            )
        return template_path

    def _load_template(self, template_path, cap):
        """Load and resize the court template image."""
        template_gray = cv2.imread(template_path, 0)
        template_color = cv2.imread(template_path)
        if template_gray is None or template_color is None:
            raise RuntimeError(f"Unable to read court template image: {template_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        template_gray = cv2.resize(template_gray, (frame_width, frame_height))
        template_color = cv2.resize(template_color, (frame_width, frame_height))
        template_match_gray = self._resize_court_match_gray(template_gray)

        return template_match_gray, template_color

    # 视频自适应模板:开场均匀采样 TEMPLATE_SAMPLE_COUNT 帧跑球场关键点模型,选有效
    # 关键点最多的帧当 is_court_view 的匹配模板。动机(生产事故 job 60aa7c7f):写死的
    # templates/demo.png 与用户视频观感差异大时(实测美网蓝场转播),0.75 阈值会把全部
    # 帧拒掉 → 零检测。用视频自己的球场帧做模板,场地颜色/光线不再影响判定,而转播
    # 素材里的回放/观众镜头依然会被正确过滤(与球场帧不像)。
    TEMPLATE_SAMPLE_COUNT = 30

    def _select_template_frame_from_video(self):
        """返回选中的 BGR 模板帧;权重缺失/无帧/全部检不出球场时返回 None(回退 demo 模板)。"""
        if not os.path.exists(self.keypoint_model_path):
            print(
                f"提示:球场关键点权重缺失({self.keypoint_model_path}),模板回退 demo 图 / "
                "Notice: court keypoint weights missing, falling back to demo template"
            )
            return None
        if self.total_frames <= 0:
            return None

        detector = CourtKeypointDetector(self.keypoint_model_path)
        cap = cv2.VideoCapture(self.video_path)
        best_frame, best_valid, best_index = None, 0, 0
        try:
            step = max(1, self.total_frames // self.TEMPLATE_SAMPLE_COUNT)
            for frame_index in range(1, self.total_frames + 1, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
                ret, frame = cap.read()
                if not ret:
                    continue
                points = detector.detect(frame)
                if points is None:
                    continue
                valid = int(np.count_nonzero(~np.isnan(points[:, 0])))
                if valid > best_valid:
                    best_frame, best_valid, best_index = frame, valid, frame_index
                    if valid == 14:
                        break
        finally:
            cap.release()

        if best_frame is None:
            print(
                "提示:采样帧均未检出球场关键点,模板回退 demo 图(若确非球场视频,"
                "后续会以 court_not_detected 收尾)/ no court keypoints in sampled frames, "
                "falling back to demo template"
            )
            return None
        print(
            f"模板帧选定: 第 {best_index} 帧({best_valid}/14 关键点)/ "
            f"template frame selected: frame {best_index} ({best_valid}/14 keypoints)"
        )
        return best_frame

    def _template_from_frame(self, frame):
        """把选中的视频帧加工成 (匹配用降采样灰度图, 全尺寸彩色图),与 _load_template 同构。"""
        template_color = frame.copy()
        template_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._resize_court_match_gray(template_gray), template_color

    def _resize_court_match_gray(self, gray_frame):
        if self.court_match_width <= 0 or gray_frame.shape[1] <= self.court_match_width:
            return gray_frame

        scale = self.court_match_width / gray_frame.shape[1]
        height = max(1, int(round(gray_frame.shape[0] * scale)))
        return cv2.resize(gray_frame, (self.court_match_width, height), interpolation=cv2.INTER_AREA)

    def _prepare_court_match_frame(self, frame, template_gray):
        target_size = (template_gray.shape[1], template_gray.shape[0])
        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _setup_video_writer(self, frame_width, frame_height, fps):

        self.temp_output_video_path = os.path.join(self.save_dir, f"temp_detect_{self.video_name}.mp4")
        

        self.video_writer = vap.setup_video_writer(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            temp_output_path=self.temp_output_video_path
        )
        
        return self.video_writer

    def _setup_court_annotation(self, template_color):
        """Set up court annotation."""

        if os.path.exists(os.path.join(self.save_dir, 'court_annotations.txt')):
            with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'r') as f:
                corners = eval(f.readline().split('=')[1])
                f.readline()
                mid_height = eval(f.readline().split('=')[1])
                roi_corners = compute_expanded_roi(corners, template_color.shape)
            self.court_detection_result = {
                "status": "cached",
                "source": "court_annotations.txt",
            }
        else:
            corners, roi_corners, mid_height = self._detect_or_annotate_court(template_color)
       
        if not corners or not roi_corners or len(corners) != 4 or len(roi_corners) != 2:
            raise RuntimeError("Court annotation is incomplete: click 4 court corners in order. ROI is generated automatically.")

        with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'w') as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
        return corners, roi_corners, mid_height

    def _detect_or_annotate_court(self, template_color):
        if self.court_detection in ('auto', 'auto-fallback'):
            detector = CourtLineAutoDetector()
            detected = detector.detect(template_color)
            candidate = detected or self._candidate_from_diagnostics(template_color, detector.last_diagnostics)
            if detected:
                preview_path = self._write_auto_court_preview(template_color, detected)
                print(
                    "自动检测到球场线: "
                    f"confidence={detected['confidence']:.2f}, "
                    f"lines={detected['line_count']}, preview={preview_path}"
                )
                # --display false(worker/headless 模式):不弹确认框阻塞等 cv2.waitKey(0)
                # 键盘输入,直接采信自动检测结果。这是 Phase 4a worker 冒烟测试发现的坑——
                # show_display 原本只控制主处理窗口,没管到这个独立的确认弹窗,导致 worker
                # 每个任务都会在这里死等键盘输入,永远跑不完。
                if not self.show_display or self._confirm_auto_court_detection(template_color, detected):
                    self.court_detection_result = {
                        "status": "auto",
                        "accepted": True,
                        "confidence": detected["confidence"],
                        "preview": preview_path,
                        "diagnostics": detected.get("diagnostics"),
                    }
                    return detected["corners"], detected["roi_corners"], detected["mid_height"]

                self.court_detection_result = {
                    "status": "manual_fallback",
                    "accepted": False,
                    "confidence": detected["confidence"],
                    "preview": preview_path,
                    "diagnostics": detected.get("diagnostics"),
                }
                print("用户拒绝自动检测结果，切换到手动四角标注。")
            elif candidate:
                preview_path = self._write_auto_court_preview(template_color, candidate)
                print(
                    "自动检测置信度偏低，已显示候选球场线: "
                    f"confidence={candidate['confidence']:.2f}, "
                    f"lines={candidate['line_count']}, preview={preview_path}"
                )
                # 跟上面 detected 分支不对称,是故意的:detected 已经过 detector 的
                # min_confidence 门槛,headless 可以放心自动采信;candidate 恰恰是
                # 「置信度不够,交互模式下本来就要弹窗人工确认」的低置信度兜底结果。
                # report.json 冻结契约里没有 confidence/status 字段,一份低置信度球场
                # 标定产出的报告跟正常报告长得一模一样——headless 下悄悄采信,等于让
                # 一份可能全错的报告蒙混过关。所以这里跟 detected 分支反着来:headless
                # 必须显式失败(置信度偏低/low confidence),不能自动接受;交互模式的
                # 人工确认弹窗原样保留不变。
                if self.show_display and self._confirm_auto_court_detection(template_color, candidate):
                    self.court_detection_result = {
                        "status": "auto_low_confidence_accepted",
                        "accepted": True,
                        "confidence": candidate["confidence"],
                        "preview": preview_path,
                        "diagnostics": candidate.get("diagnostics"),
                    }
                    return candidate["corners"], candidate["roi_corners"], candidate["mid_height"]

                if not self.show_display:
                    self.court_detection_result = {
                        "status": "auto_low_confidence_rejected",
                        "accepted": False,
                        "confidence": candidate["confidence"],
                        "preview": preview_path,
                        "diagnostics": candidate.get("diagnostics"),
                    }
                    raise RuntimeError(
                        "Court auto-detection confidence is too low (自动检测置信度偏低, "
                        f"confidence={candidate['confidence']:.2f}) and --display is false "
                        "(headless/worker mode): refusing to silently accept a low-confidence "
                        "court calibration. Re-record with clearer/fully-visible court lines, "
                        "or run interactively once (--display true) to confirm manually."
                    )

                self.court_detection_result = {
                    "status": "manual_fallback",
                    "accepted": False,
                    "confidence": candidate["confidence"],
                    "preview": preview_path,
                    "diagnostics": candidate.get("diagnostics"),
                }
                print("用户拒绝低置信度自动检测结果，切换到手动四角标注。")

            if self.court_detection == 'auto':
                if self.court_detection_result is None:
                    self.court_detection_result = {
                        "status": "auto",
                        "accepted": False,
                        "diagnostics": detector.last_diagnostics,
                    }
                raise RuntimeError(
                    "Court auto-detection failed: "
                    f"{detector.last_diagnostics}. Use --court-detection auto-fallback or manual."
                )

            if self.court_detection_result is None:
                print(f"自动球场线检测失败，切换到手动四角标注。diagnostics={detector.last_diagnostics}")
                self.court_detection_result = {
                    "status": "manual_fallback",
                    "accepted": False,
                    "diagnostics": detector.last_diagnostics,
                }

        if not self.show_display:
            # 同上:headless 模式没有鼠标可点 4 个角点,annotate_court() 会开窗口
            # 死等鼠标事件。auto-detection 完全没结果时没有退路,只能显式失败,
            # 不能悄悄挂起——worker 侧靠这个异常上报 error_code 给 backend。
            raise RuntimeError(
                "Court annotation needs interactive manual corner-clicking but "
                "--display is false (headless/worker mode), and auto-detection did "
                "not produce a usable result for this footage."
            )
        corners, roi_corners, mid_height = annotate_court(template_color)
        if self.court_detection_result is None:
            self.court_detection_result = {
                "status": "manual",
                "accepted": True,
            }
        return corners, roi_corners, mid_height

    def _candidate_from_diagnostics(self, template_color, diagnostics):
        if not diagnostics or not diagnostics.get("corners"):
            return None

        corners = [(int(x), int(y)) for x, y in diagnostics["corners"]]
        roi_corners = compute_expanded_roi(corners, template_color.shape)
        mid_height = int((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4)
        return {
            "corners": corners,
            "roi_corners": roi_corners,
            "mid_height": mid_height,
            "line_count": int(diagnostics.get("line_count", 0)),
            "confidence": float(diagnostics.get("confidence", 0.0)),
            "diagnostics": diagnostics,
        }

    def _confirm_auto_court_detection(self, template_color, detected):
        preview = self._build_auto_court_preview(template_color, detected)
        cv2.putText(
            preview,
            "Enter/Y: accept auto  M/R/Esc: manual corners",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        window_name = "Auto court detection preview"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, preview)
        print("请检查自动球场线预览：按 Enter/Y 接受，按 M/R/Esc 改为手动四角标注。")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10, ord('y'), ord('Y')):
                cv2.destroyWindow(window_name)
                return True
            if key in (27, ord('m'), ord('M'), ord('r'), ord('R')):
                cv2.destroyWindow(window_name)
                return False

    def _write_auto_court_preview(self, template_color, detected):
        preview = self._build_auto_court_preview(template_color, detected)
        preview_path = os.path.join(self.save_dir, "auto_court_preview.png")
        cv2.imwrite(preview_path, preview)
        return preview_path

    def _build_auto_court_preview(self, template_color, detected):
        preview = template_color.copy()
        try:
            preview, _ = CourtMapper(detected["corners"]).draw_court_overlay(preview)
        except Exception:
            pass
        corners = np.array(detected["corners"], dtype=np.int32)
        cv2.polylines(preview, [corners], True, (0, 255, 255), 3, cv2.LINE_AA)
        for index, point in enumerate(detected["corners"], start=1):
            cv2.circle(preview, tuple(point), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(preview, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        reference_score = detected.get("diagnostics", {}).get("reference_score")
        reference_text = f", ref={reference_score:.2f}" if isinstance(reference_score, (int, float)) else ""
        cv2.putText(
            preview,
            f"auto court confidence={detected['confidence']:.2f}{reference_text}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def _release_capture_resources(self, cap):
        """只释放读写资源,不跑任何后处理(零球场帧提前收尾与 _cleanup 共用)。"""
        if self.detection_writer is not None:
            self.detection_writer.close()
            self.detection_writer = None

        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            time.sleep(1)

        cap.release()

    def _cleanup(self, cap):
        """Clean up resources and merge audio when needed."""
        self._release_capture_resources(cap)

        if self.show_bounce_detection and self.bounce_detector is not None:
            self._finalize_bounce_detection()

        # Task 7：阶段 2 shot_metrics 写盘之后追加比赛层（behind --match-scoring）。
        if self.enable_match_scoring:
            self._run_match_layer()

        if self.show_display:
            cv2.destroyAllWindows()

        if hasattr(self, 'keep_audio') and self.keep_audio:
            vap.process_video_with_audio(
                video_path=self.video_path,
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path,
                save_dir=self.save_dir
            )
        else:
            vap.process_video_without_audio(
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path
            )

    def _finalize_bounce_detection(self):
        if not os.path.exists(self.detections_path):
            return

        events = self.bounce_detector.process_detections(
            self.detections_path,
            output_path=self.bounce_events_path,
            trajectory_output_path=self.cleaned_ball_trajectory_path,
            rewrite_detections=True,
        )
        print(f"弹跳后处理完成: {len(events)} 个候选点，结果={self.bounce_events_path}")

        shot_metrics_entries, bounce_line_calls = self._run_shot_and_line_call_pipeline(events)
        self._patch_metadata_with_camera(self.camera_dict)

        if not self.temp_output_video_path or not os.path.exists(self.temp_output_video_path):
            return

        # 真回合切分给标注视频画「回合: N」——第一遍的镜头切换计数对固定机位恒为 1
        # (误导),这里用与比赛层同源的 extract_rallies 口径重算(比赛层要到
        # annotate 之后才跑,顺序不能倒:它依赖本函数写盘的 shot_metrics/line_call;
        # 这几步是纯内存计算,重复算一遍代价可忽略)。失败只降级不画,不阻断标注。
        rally_spans = None
        try:
            from .analysis.match_layer import attach_shot_types, build_bounces, build_visible
            from .analysis.rally import extract_rallies

            detections_by_frame = {
                int(record["frame"]): record
                for record in self._load_detection_records()
                if "frame" in record
            }
            shots_with_type = attach_shot_types(
                shot_metrics_entries, detections_by_frame, self.fps,
                upper_hand=self.upper_hand, lower_hand=self.lower_hand,
            )
            rallies = extract_rallies(
                shots_with_type,
                build_bounces(detections_by_frame),
                build_visible(detections_by_frame, total_frames=self.total_frames),
                self.fps,
            )
            if rallies:
                rally_spans = [(r.start_frame, r.end_frame) for r in rallies]
        except Exception as exc:  # noqa: BLE001 - 回合数只是叠加显示,不值得让标注失败
            print(f"警告：回合切分失败，标注视频不画回合数 / rally split failed, skipping rally overlay: {exc}")

        rally_label_pos, rally_label_scale = self.stats_visualizer.rally_label_geometry()
        annotated_path = os.path.join(self.save_dir, f"temp_bounce_{self.video_name}.mp4")
        self.bounce_detector.annotate_video(
            self.temp_output_video_path,
            annotated_path,
            events,
            trajectory_points=self.bounce_detector.processed_points,
            draw_minimap_bounces=self.show_mini_map,
            draw_processed_trajectory=self.show_tennis_ball_trajectory,
            bounce_line_calls=bounce_line_calls,
            shot_hits={entry["hit_frame"]: entry for entry in shot_metrics_entries},
            rally_spans=rally_spans,
            rally_label_pos=rally_label_pos,
            rally_label_font_size=max(8, int(rally_label_scale * 30)),
        )
        if os.path.exists(annotated_path):
            os.replace(annotated_path, self.temp_output_video_path)

    # ------------------------------------------------------------------
    # Task 7: 比赛层（shot_type/rally/point_outcome/scoring/stats/highlights）主流程接线
    # ------------------------------------------------------------------

    def _run_match_layer(self):
        """比赛层全链路，整体包在一个 try/except 里：任何一步异常都只打印双语警告
        并跳过比赛层，绝不影响已经写盘的阶段 2 输出（shot_metrics.json 等）。"""
        try:
            if not self.enable_shot_metrics or not os.path.exists(self.shot_metrics_path):
                print(
                    "提示：--match-scoring 需要 --shot-metrics true 且已生成 shot_metrics.json，"
                    "已跳过比赛层分析 / "
                    "Notice: --match-scoring requires --shot-metrics true and an existing "
                    "shot_metrics.json, skipping match layer analysis"
                )
                return
            if not os.path.exists(self.detections_path):
                print(
                    "提示：缺少 detections.jsonl，已跳过比赛层分析 / "
                    "Notice: detections.jsonl missing, skipping match layer analysis"
                )
                return

            with open(self.shot_metrics_path, "r", encoding="utf-8") as file:
                shot_metrics_entries = json.load(file)

            detections_by_frame = {
                int(record["frame"]): record
                for record in self._load_detection_records()
                if "frame" in record
            }

            result = run_match_layer(
                shot_metrics_entries, detections_by_frame, self.fps,
                first_server=self.first_server,
                upper_hand=self.upper_hand, lower_hand=self.lower_hand,
                sets_to_win=self.sets_to_win, no_ad=self.no_ad,
                total_frames=self.total_frames,
            )

            # shot_metrics.json 是阶段 2 已经写盘的真产物，这里只是给每条追加
            # "shot_type" 后整体重写；用 tmp + os.replace（与 _inject_line_call
            # 同一套原子替换写法）而不是 write_json 的直接 open("w") 截断写，
            # 避免序列化中途失败时把阶段 2 的好文件截断成半截坏文件。
            self._atomic_write_json(self.shot_metrics_path, result["shot_metrics"])
            self._atomic_write_json(self.match_score_path, result["match_score"])
            self._atomic_write_json(self.match_stats_path, result["match_stats"])
            print(
                f"比赛层分析完成：{len(result['points'])} 分，"
                f"结果={self.match_score_path} / {self.match_stats_path} / "
                f"Match layer analysis complete: {len(result['points'])} points, "
                f"output={self.match_score_path} / {self.match_stats_path}"
            )

            if self.enable_highlights:
                self._export_highlights(result)
        except Exception as exc:  # noqa: BLE001 - 比赛层失败不能影响阶段 2 产物
            print(
                "警告：比赛层分析失败，已跳过（阶段 2 输出不受影响）/ "
                f"Warning: match layer analysis failed, skipping (phase-2 outputs unaffected): {exc}"
            )

    @staticmethod
    def _atomic_write_json(path, payload):
        """先写到 `{path}.tmp` 再 `os.replace`（与 `_inject_line_call` 同一套原子
        替换写法），避免 `write_json` 的直接 open("w") 截断写在序列化中途失败时
        把目标路径已有的旧文件（例如阶段 2 的 shot_metrics.json）截断成半截坏文件。"""
        tmp_path = f"{path}.tmp"
        write_json(tmp_path, payload)
        os.replace(tmp_path, path)

    def _export_highlights(self, match_layer_result):
        """从 match layer 结果里挑集锦回合并导出；export_highlights 本身已对
        ffmpeg 缺失/drawtext 缺失做了降级，这里额外兜底一层，防止选段/映射逻辑
        本身出错波及比赛层已写盘的 JSON 产物。"""
        try:
            rallies = match_layer_result["rallies"]
            points = match_layer_result["points"]
            rally_score_lines = match_layer_result["rally_score_lines"]

            selected_rallies = select_highlight_rallies(rallies, points)
            index_by_id = {id(rally): index for index, rally in enumerate(rallies)}
            selected_score_lines = [
                rally_score_lines[index_by_id[id(rally)]] for rally in selected_rallies
            ]
            export_highlights(
                self.video_path, selected_rallies, selected_score_lines,
                self.highlights_path, self.fps,
            )
        except Exception as exc:  # noqa: BLE001 - 集锦导出失败不能影响比赛层其余产物
            print(
                "警告：集锦导出失败，已跳过 / "
                f"Warning: highlight export failed, skipping: {exc}"
            )

    # ------------------------------------------------------------------
    # Task 10: shot metrics / spin / line calling 主流程接线
    # ------------------------------------------------------------------

    # 回合边界判定阈值：detections.jsonl 只在 is_court=True 的帧写记录（见
    # _process_frame 的 `if not is_court: return`），同一回合内偶发 1-2 帧误判
    # 非球场只会留下小缝隙；真正的回合切换需要连续 >= non_court_frames_threshold
    # (=5) 帧非球场才会翻转 rally_active，此后摄像机通常会离场更久。用大于该
    # 阈值的间隔帧数作为回合分界，兼顾"容忍瞬时误判"与"识别真实回合切换"。
    RALLY_GAP_FRAMES = 15

    def _run_shot_and_line_call_pipeline(self, events):
        """bounce line_call 回填 detections.jsonl + 相机标定 + 逐回合
        segments/metrics/spin/line_call -> shot_metrics.json。

        返回 (shot_metrics_entries, bounce_line_calls)，供最终视频叠加层复用
        （避免重复读盘 / 重复调用 call_bounce）。
        """
        records = self._load_detection_records()

        bounce_line_calls = {}
        if self.line_call_mode:
            for event in events:
                court = event.get("court")
                if court is None:
                    continue
                verdict, _distance = call_bounce(court, mode=self.line_call_mode)
                bounce_line_calls[int(event["frame"])] = verdict
            self._inject_line_call(records, bounce_line_calls)
        else:
            print("提示：--line-call off，跳过弹跳判罚 / Notice: --line-call off, skipping bounce line calling")

        shot_metrics_entries = []
        camera_dict = None
        if self.enable_shot_metrics:
            rallies = self._split_records_into_rallies(records)
            camera_spans, recalibrated_at_frames = self._calibrate_camera()

            points_by_frame = {
                int(point.frame): {
                    "frame": int(point.frame),
                    "time_sec": point.time_sec,
                    "image": point.image,
                    "court": point.court,
                }
                for point in (self.bounce_detector.processed_points or [])
            }
            whole_video_positions = self._median_positions(records)

            for rally in rallies:
                rally_points = [points_by_frame[frame] for frame in rally["frames"] if frame in points_by_frame]
                rally_bounces = [event for event in events if rally["start"] <= int(event["frame"]) <= rally["end"]]
                rally_positions = self._median_positions(rally["records"])
                resolved_positions, missing_sides = self._resolve_player_positions(rally_positions, whole_video_positions)
                for side in missing_sides:
                    print(
                        f"警告：第 {rally['start']}-{rally['end']} 帧回合缺少{side}方球员位置数据，"
                        f"跳过该方击球分析 / Warning: no player position data for '{side}' side in "
                        f"rally frames {rally['start']}-{rally['end']}, skipping that side's shots"
                    )
                # R12：一个回合可能横跨标定 span 边界（罕见——漂移重标定发生在回合中间），
                # 用回合中点帧去查该回合该用哪一段相机标定。
                rally_mid_frame = (rally["start"] + rally["end"]) // 2
                camera = self._camera_for_frame(camera_spans, rally_mid_frame)
                entries = compute_shot_metrics_entries(
                    rally_points, rally_bounces, resolved_positions, self.fps, camera, self.line_call_mode
                )
                if missing_sides:
                    entries = [entry for entry in entries if entry["hitter"] not in missing_sides]
                shot_metrics_entries.extend(entries)

            write_shot_metrics(self.shot_metrics_path, shot_metrics_entries)
            print(f"击球指标计算完成: {len(shot_metrics_entries)} 拍，结果={self.shot_metrics_path}")

            if camera_spans:
                # R12：metadata.json 的 camera 只存最后一段标定（+完整的重标定帧号列表）。
                camera_dict = camera_spans[-1]["camera"].to_dict()
                camera_dict["recalibrated_at_frames"] = recalibrated_at_frames
        else:
            print("提示：--shot-metrics false，跳过击球速度/旋转分析 / Notice: --shot-metrics false, skipping shot speed/spin analysis")

        self.camera_dict = camera_dict
        return shot_metrics_entries, bounce_line_calls

    def _load_detection_records(self):
        records = []
        with open(self.detections_path, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _inject_line_call(self, records, bounce_line_calls):
        """把 bounce_line_calls 回填进（内存中的）records 的 bounce.line_call，再整体重写
        detections.jsonl——records 由调用方复用，避免为 line_call 单独多一趟读盘。"""
        for record in records:
            bounce = record.get("bounce")
            if bounce is not None:
                bounce["line_call"] = bounce_line_calls.get(int(record.get("frame", -1)))

        tmp_path = f"{self.detections_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                file.write("\n")
        os.replace(tmp_path, self.detections_path)

    def _split_records_into_rallies(self, records):
        records = sorted(records, key=lambda record: int(record.get("frame", 0)))
        chunks = []
        current = []
        prev_frame = None
        for record in records:
            frame = int(record.get("frame", 0))
            if prev_frame is not None and frame - prev_frame > self.RALLY_GAP_FRAMES:
                chunks.append(current)
                current = []
            current.append(record)
            prev_frame = frame
        if current:
            chunks.append(current)

        rallies = []
        for chunk in chunks:
            frames = [int(record.get("frame", 0)) for record in chunk]
            rallies.append({"records": chunk, "frames": frames, "start": frames[0], "end": frames[-1]})
        return rallies

    def _median_positions(self, records):
        """{"upper": [x,y]|None, "lower": [x,y]|None}：该侧无任何 court 数据时为 None（R3）。"""
        collected = {"upper": [], "lower": []}
        for record in records:
            players = record.get("players") or {}
            for side in ("upper", "lower"):
                court = (players.get(side) or {}).get("court")
                if court is not None:
                    collected[side].append(court)

        result = {}
        for side, points in collected.items():
            if not points:
                result[side] = None
                continue
            array = np.array(points, dtype=float)
            result[side] = [float(np.median(array[:, 0])), float(np.median(array[:, 1]))]
        return result

    def _resolve_player_positions(self, rally_positions, whole_video_positions):
        """R3：回合内位置缺失 -> 退化到全视频中位数；全视频也没有 -> 该侧标记 missing，
        由调用方过滤掉对应 hitter 的 shot_metrics 条目（占位坐标只为让 extract_segments
        不崩，最终会被丢弃，数值本身不参与任何保留下来的输出）。"""
        resolved = {}
        missing = []
        for side in ("upper", "lower"):
            value = rally_positions.get(side) or whole_video_positions.get(side)
            if value is None:
                missing.append(side)
                value = [5.485, 0.0] if side == "upper" else [5.485, 23.77]
            resolved[side] = value
        return resolved, missing

    # 固定机位标定：视频前 CAMERA_INITIAL_SAMPLE_FRAMES 帧，每 CAMERA_INITIAL_SAMPLE_STEP
    # 帧抽 1 帧检测 -> 逐点中位数 -> CameraModel.calibrate，整段视频复用。
    CAMERA_INITIAL_SAMPLE_FRAMES = 300
    CAMERA_INITIAL_SAMPLE_STEP = 10
    # 某关键点在采样帧中有效（非 NaN）次数 < 此值 -> 该点整体判无效（见 camera_calibration.py）。
    CAMERA_KEYPOINT_MIN_VALID_SAMPLES = 3
    CAMERA_MIN_VALID_KEYPOINTS = 6
    CAMERA_MAX_REPROJECTION_ERROR_PX = 15.0
    # 漂移守卫：标定完成后每 CAMERA_DRIFT_CHECK_INTERVAL_FRAMES（约 30s@30fps）帧重检 1 帧；
    # 命中漂移则用其后 CAMERA_RECALIBRATION_WINDOW_FRAMES 帧（同样每 10 帧抽 1 帧）重新标定。
    # 单关键点重投影误差超过此值即视为稳定检错的外点，剔除后重标定
    CAMERA_KEYPOINT_OUTLIER_THRESHOLD_PX = 10.0
    CAMERA_DRIFT_CHECK_INTERVAL_FRAMES = 900
    CAMERA_DRIFT_THRESHOLD_PX = 10.0
    CAMERA_DRIFT_MIN_KEYPOINTS = 4
    CAMERA_RECALIBRATION_WINDOW_FRAMES = 300

    def _calibrate_camera(self):
        """固定机位相机标定 + 漂移守卫（Task 10 amended brief）。

        权重缺失 / --court-calibration homography 任一命中，或初始标定失败（关键点检测
        全军覆没 / 有效关键点 < 6 / 重投影误差 > 15px），整体降级为 homography-only：
        返回 ([], [])，shot_metrics 只保留 line_call。

        标定成功后，返回 (spans, recalibrated_at_frames)：
            spans: 按帧区间排序的 [{"start_frame", "end_frame", "camera"}, ...]，
                元素之间首尾相接覆盖 [1, total_frames]（1-indexed，与 detections.jsonl /
                rally["start"]/["end"] 的 frame 计数一致，见 _process_frame 的
                `frame_count += 1`）；漂移守卫每命中一次就在此追加一段。
            recalibrated_at_frames: 实际触发了重标定（且重标定成功）的帧号列表（1-indexed）；
                未发生漂移重标定时为空列表。
        """
        if self.court_calibration == 'homography':
            print(
                "提示：--court-calibration homography 强制单应性降级，跳过相机标定 / "
                "Notice: --court-calibration homography forces homography-only degrade, "
                "skipping camera calibration"
            )
            return [], []

        if not os.path.exists(self.keypoint_model_path):
            print(
                f"警告：球场关键点权重缺失（{self.keypoint_model_path}），降级为单应性模式 / "
                f"Warning: court keypoint weights missing ({self.keypoint_model_path}), "
                "degrading to homography-only mode"
            )
            return [], []

        if self.total_frames <= 0:
            print("提示：视频无有效帧，跳过相机标定 / Notice: video has no frames, skipping camera calibration")
            return [], []

        detector = CourtKeypointDetector(self.keypoint_model_path)
        cap = cv2.VideoCapture(self.video_path)
        try:
            initial_detections = self._sample_keypoint_detections(
                cap, detector, start_frame=1,
                span_frames=self.CAMERA_INITIAL_SAMPLE_FRAMES, step=self.CAMERA_INITIAL_SAMPLE_STEP,
            )
            if not initial_detections:
                print(
                    "警告：关键点检测在所有采样帧均失败，降级为单应性模式 / "
                    "Warning: keypoint detection failed on all sampled frames, degrading to homography-only mode"
                )
                return [], []

            baseline_median, baseline_mask = median_keypoints_over_frames(
                initial_detections, min_valid_frames=self.CAMERA_KEYPOINT_MIN_VALID_SAMPLES
            )
            camera = self._try_calibrate_from_median(baseline_median, baseline_mask)
            if camera is None:
                return [], []

            spans = [{"start_frame": 1, "end_frame": self.total_frames, "camera": camera}]
            recalibrated_at_frames = []

            next_check = 1 + self.CAMERA_DRIFT_CHECK_INTERVAL_FRAMES
            while next_check <= self.total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_check - 1)  # POS_FRAMES 是 0-indexed
                ret, frame = cap.read()
                if ret:
                    current_points = detector.detect(frame)
                    if current_points is not None and keypoints_drifted(
                        current_points, baseline_median, baseline_mask,
                        threshold_px=self.CAMERA_DRIFT_THRESHOLD_PX,
                        min_keypoints=self.CAMERA_DRIFT_MIN_KEYPOINTS,
                    ):
                        print(
                            f"警告：检测到第 {next_check} 帧附近相机可能被碰动，正在用其后 "
                            f"{self.CAMERA_RECALIBRATION_WINDOW_FRAMES} 帧重新标定 / "
                            f"Warning: possible camera bump detected near frame {next_check}, "
                            f"recalibrating using the next {self.CAMERA_RECALIBRATION_WINDOW_FRAMES} frames"
                        )
                        recal_detections = self._sample_keypoint_detections(
                            cap, detector, start_frame=next_check,
                            span_frames=self.CAMERA_RECALIBRATION_WINDOW_FRAMES,
                            step=self.CAMERA_INITIAL_SAMPLE_STEP,
                        )
                        new_camera = None
                        if recal_detections:
                            new_median, new_mask = median_keypoints_over_frames(
                                recal_detections, min_valid_frames=self.CAMERA_KEYPOINT_MIN_VALID_SAMPLES
                            )
                            new_camera = self._try_calibrate_from_median(new_median, new_mask)
                        if new_camera is not None:
                            spans[-1]["end_frame"] = next_check - 1
                            spans.append({"start_frame": next_check, "end_frame": self.total_frames, "camera": new_camera})
                            recalibrated_at_frames.append(int(next_check))
                            baseline_median, baseline_mask = new_median, new_mask
                        else:
                            print(
                                "警告：重标定失败，继续沿用上一段相机标定 / "
                                "Warning: recalibration failed, continuing with the previous camera calibration"
                            )
                next_check += self.CAMERA_DRIFT_CHECK_INTERVAL_FRAMES

            return spans, recalibrated_at_frames
        finally:
            cap.release()

    def _far_roi_rect_for_ball(self):
        """远场 ROI 裁剪矩形（懒计算 + 缓存）：远端两底线角（image_court_corners
        前两点，世界 y=0 底线）+ 球网两端图像坐标 交给 compute_far_roi_rect。
        开关关闭 / 非 tracknet 后端 / 尚无球场标定 / 矩形退化时返回 None
        （detect_ball 收到 None 即退回单次全帧推理）。球场角点整段视频固定
        （手动/自动标注一次），矩形只算一次。"""
        if not getattr(self, 'far_roi', False) or self.ball_detector != 'tracknet':
            return None
        if getattr(self, '_far_roi_rect_computed', False):
            return self._far_roi_rect_cache
        mapper = getattr(self, 'court_mapper', None)
        if mapper is None:
            return None  # 标定尚未就绪时不落缓存，下次再试
        net_y = mapper.court_dimensions[1] / 2
        net_left = mapper.court_to_image((0.0, net_y))
        net_right = mapper.court_to_image((mapper.court_dimensions[0], net_y))
        far_points = [
            tuple(mapper.image_court_corners[0]),
            tuple(mapper.image_court_corners[1]),
            tuple(net_left),
            tuple(net_right),
        ]
        self._far_roi_rect_cache = compute_far_roi_rect(far_points, self.frame_width, self.frame_height)
        self._far_roi_rect_computed = True
        return self._far_roi_rect_cache

    def _sample_keypoint_detections(self, cap, detector, start_frame, span_frames, step):
        """在 1-indexed 帧号区间 [start_frame, min(start_frame+span_frames-1, total_frames)]
        内每 step 帧抽 1 帧跑关键点检测，返回检测结果不为 None 的 np.ndarray[14,2] 列表
        （复用同一个 cap；cv2.CAP_PROP_POS_FRAMES 是 0-indexed，故 seek 时 -1）。"""
        detections = []
        end_frame_inclusive = min(start_frame + span_frames - 1, self.total_frames)
        for frame_index in range(start_frame, end_frame_inclusive + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            ret, frame = cap.read()
            if not ret:
                continue
            points = detector.detect(frame)
            if points is not None:
                detections.append(points)
        return detections

    def _try_calibrate_from_median(self, median_points, valid_mask):
        """有效关键点 < 6 或重投影误差 > 15px 时打印双语警告并返回 None（调用方按此降级）。"""
        valid_count = int(valid_mask.sum())
        if valid_count < self.CAMERA_MIN_VALID_KEYPOINTS:
            print(
                f"警告：有效关键点仅 {valid_count} 个（<6），降级为单应性模式 / "
                f"Warning: only {valid_count} valid keypoints (<6), degrading to homography-only mode"
            )
            return None

        image_points = median_points[valid_mask]
        world_points = COURT_KEYPOINTS_M[valid_mask]
        # 带外点剔除的标定（CourtCheck 重投影误差择优思想）：某关键点被检测
        # 模型稳定检错时（中位数救不了），全点一把梭会把单应整体拉偏 → 所有
        # 落点系统性偏移。剔除保底 CAMERA_MIN_VALID_KEYPOINTS 个点；15px
        # 整体门限只看保留下来的内点。
        camera, inlier_mask = calibrate_with_outlier_rejection(
            image_points, world_points, (self.frame_width, self.frame_height),
            point_error_threshold_px=self.CAMERA_KEYPOINT_OUTLIER_THRESHOLD_PX,
            min_points=self.CAMERA_MIN_VALID_KEYPOINTS,
        )
        rejected_count = int((~inlier_mask).sum())
        if rejected_count:
            valid_count = int(inlier_mask.sum())
            print(
                f"提示：{rejected_count} 个关键点重投影误差过大被剔除，剩 {valid_count} 个参与标定 / "
                f"Notice: rejected {rejected_count} keypoint(s) with excessive reprojection error, "
                f"{valid_count} keypoint(s) used for calibration"
            )
        error = camera.reprojection_error(image_points[inlier_mask], world_points[inlier_mask])
        if error > self.CAMERA_MAX_REPROJECTION_ERROR_PX:
            print(
                f"警告：重投影误差 {error:.1f}px > 15px，降级为单应性模式 / "
                f"Warning: reprojection error {error:.1f}px > 15px, degrading to homography-only mode"
            )
            return None

        print(
            f"相机标定成功：{valid_count} 个关键点，重投影误差 {error:.2f}px / "
            f"Camera calibrated: {valid_count} keypoints, reprojection error {error:.2f}px"
        )
        return camera

    @staticmethod
    def _camera_for_frame(spans, frame):
        """R12：一个回合可能横跨标定 span 边界，用该回合的代表帧（调用方传中点帧）
        查找覆盖它的 CameraModel；spans 为空（完全降级）时返回 None。"""
        for span in spans:
            if span["start_frame"] <= frame <= span["end_frame"]:
                return span["camera"]
        return spans[-1]["camera"] if spans else None

    def _patch_metadata_with_camera(self, camera_dict):
        if not os.path.exists(self.metadata_path):
            return
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["camera"] = camera_dict
        write_json(self.metadata_path, metadata)

    def analyze_tennis_ball(self, roi_corners, corners):
        """Hit-point analysis is currently disabled."""
        raise RuntimeError(
            "Hit-point analysis is disabled until it is migrated to detections.jsonl."
        )

    def is_court_view(self, frame, template_gray, threshold=0.75):
        """Return whether the frame matches the court template."""
        result = cv2.matchTemplate(frame, template_gray, cv2.TM_CCOEFF_NORMED)
        # print("match score: ", result)
        return np.max(result) >= threshold

    def draw_court_roi(self, frame, corners, roi_corners):
        self.court_mapper = CourtMapper(corners)
        overlay, mid_height_int = self.court_mapper.draw_court_overlay(frame)
        cv2.rectangle(overlay, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
        return overlay


