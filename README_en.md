# RallyVision: AI Tennis Match Analysis Assistant 🎾

<div align="center">

[![GitHub license](https://img.shields.io/github/license/EricXue92/RallyVision)](https://github.com/EricXue92/RallyVision/blob/main/LICENSE)

**A computer-vision-based tennis match video analysis tool**

Built upon [Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis) (Apache 2.0)

[Chinese](README.md) | [English](README_en.md)

</div>

### 🎬 Video Analysis Results

| RTMPose Pose Detection                                            | YOLO26s Person Detection                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![RTMPose pose detection demo](assets/en_rtmpose_detect_demo.gif) | ![YOLO26s person detection demo](assets/en_yolo26s_detect_demo.gif) |

## 📝 Changelog

- **2026-08-22**: Phase-2 capabilities shipped — fixed-camera calibration (`--court-calibration`), shot speed/spin estimation (`--shot-metrics`), bounce IN/OUT line calling (`--line-call`), a new WASB-SBDT ball-detection backend (`--ball-detector wasb`, alongside yolo/tracknet), and a real-data speed validation tool `tools/validate_speed.py`.
- **2026-06-22**: Organized the open-source README and added tennis ball bounce detection.
- **Current version**: Supports player detection, tennis ball detection, court coordinate mapping, trajectory statistics, rally detection, mini-map overlays, heatmaps/scatter plots, and annotated video output.
- **Experimental features**: Automatic outer court corner detection and tennis ball bounce detection are still being improved, and are suitable for research and further development.

## 🗺️ Roadmap

- [x] Frame-by-frame tennis match video analysis
- [x] YOLO person detection and multiple pose model options
- [x] YOLO tennis ball detection integration
- [x] Manual/automatic court annotation and court coordinate mapping
- [x] Player movement trajectories, speed, distance, and rally statistics
- [x] Tennis ball trajectory and bounce point annotation
- [x] Standard tennis court mini-map overlay
- [x] Chinese / English visualization text
- [x] Heatmaps, scatter plots, and detection data export
- [x] Fixed-camera calibration with drift detection and auto re-calibration
- [x] Shot speed and spin estimation from 3D trajectory fitting
- [x] Bounce point IN/OUT line calling
- [x] Additional ball-detection backends (TrackNet, WASB-SBDT)
- [x] Real-data validation tooling (`tools/validate_speed.py`)
- [ ] More stable tennis ball bounce point recognition
- [ ] More accurate tennis ball detection model
- [ ] Batch video analysis workflow

---

## ✨ Features

- **Player detection** - Uses YOLO person bounding boxes by default, and can also switch to RTMPose, RTMO, or Ultralytics YOLO Pose for pose estimation.
- **Tennis ball detection** - Three selectable backends (`--ball-detector yolo|tracknet|wasb`): YOLO single-frame box detection, TrackNet multi-frame heatmap, or WASB-SBDT multi-frame heatmap; raw detections are written to data files, while the final video draws the cleaned trajectory after post-processing, filtering, and interpolation.
- **Court annotation** - Tries to automatically detect the four outer doubles court corners by default, and falls back to manual corner clicks if detection fails.
- **Court coordinate mapping** - Maps image coordinates to a standard doubles tennis court coordinate system modeled as `10.97m x 23.77m`.
- **Camera calibration** - `--court-calibration keypoints` (default) uses 14-point court keypoint detection to run a full fixed-camera calibration (intrinsics + extrinsics) for shot speed/spin estimation; `--court-calibration homography` degrades to homography-only mapping (no speed/spin, line calling only). If the camera is bumped mid-recording, drift is auto-detected and the camera is re-calibrated (`metadata.json` records `recalibrated_at_frames`).
- **Shot speed and spin** - `--shot-metrics true` (default) fits each shot's 3D ball trajectory against the calibrated camera and physics model, computing per-shot speed (km/h) and spin direction (topspin/slice/flat). Segments where the fit fails (`fit_ok=false`) are kept in the output with `null` values rather than silently dropped.
- **Bounce line calling** - `--line-call singles|doubles|off` calls each bounce point IN/OUT against singles or doubles sidelines, with a close-call tolerance band.
- **Player position tracking** - Records player court coordinates, movement trajectories, speed, and distance.
- **Rally detection** - Automatically detects rally start/end from consecutive court-view frames, and records rally IDs in both the video overlay and detection data.
- **Bounce point detection** - After video processing, the full tennis ball trajectory is cleaned, interpolated, and scored by rules by default; the cleaned ball trajectory and bounce points are drawn on the main frame and mini-map.
- **Mini-map overlay** - Displays a standard tennis court mini-map in the output video, with player, ball, and bounce point positions.
- **Position charts** - Automatically generates player position heatmaps and scatter plots.
- **Chinese / English display** - Visualization text can be switched with `--language zh/en`.
- **Local processing** - Videos, models, and analysis results are all stored locally.

### 📊 Court and Position Visualizations

| Automatic Court Detection                     | Player Position Heatmap                                | Player Position Scatter Plot                                |
| --------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| ![Automatic court detection](assets/auto.png) | ![Player position heatmap](assets/en_demo_heatmap.png) | ![Player position scatter plot](assets/en_demo_scatter.png) |

## 🧩 Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python version and dependencies automatically)
- FFmpeg added to the system `PATH`
- OpenCV / PyTorch / Ultralytics / RTMLib / ONNX Runtime
- NVIDIA GPU recommended; CPU execution works, but video analysis will be significantly slower

## ⚙️ Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. After installing uv, run in the project root:

```bash
uv sync
```

uv automatically downloads a suitable Python version, creates `.venv`, and installs all dependencies (works on Windows / Linux / macOS).

### GPU Acceleration (Windows / NVIDIA)

The default dependencies use CPU builds of PyTorch and ONNX Runtime. For GPU acceleration, first confirm:

- NVIDIA GPU driver is installed and `nvidia-smi` works.
- CUDA 12.1 PyTorch wheels are recommended.
- If DLL loading fails, install or repair Microsoft Visual C++ Redistributable 2015-2022 x64.

PowerShell:

```bash
uv pip uninstall torch torchvision onnxruntime
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
uv pip install onnxruntime-gpu==1.20.1
```

Verify GPU availability:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected output includes:

```text
cuda: True
CUDAExecutionProvider
```

Switch back to CPU builds:

```bash
uv sync --reinstall
```

## 🧠 Model Weights

Before the first run, download the model weights from the project's GitHub Release page:

```text
https://github.com/yo-WASSUP/Good-Tennis/releases/latest
```

Put all downloaded weight files into the `weights/` folder under the project root, and keep the following default paths unchanged:

```text
weights/tennis-ball.pt
weights/yolo26s.pt
weights/yolo11s-pose.pt
weights/yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx
weights/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx
weights/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx
```

If a default weight file is missing, the program will report the missing file at startup. You can also pass custom model paths with `--ball-model`, `--person-model`, and `--yolo-pose-model`.

Tennis ball detection reads the tennis ball YOLO weight by default (`--ball-detector yolo`):

```text
weights/tennis-ball.pt
```

Switching to `--ball-detector tracknet` requires `weights/tracknet_ball.pt`; switching to `--ball-detector wasb` requires `weights/wasb_tennis.pth` (from [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT), MIT License — download instructions in `weights/README.md`).

Shot speed/spin estimation (`--shot-metrics`) with `--court-calibration keypoints` (the default) requires `weights/court_keypoints.pt` for 14-point court keypoint detection. If the weight is missing, or calibration fails (reprojection error > 15px or fewer than 6 valid keypoints), the program prints a bilingual warning and automatically degrades to homography-only (equivalent to `--court-calibration homography`) — in that mode only line calling is available, no speed/spin.

The player detection model is selected with `--player-detector`. The default is `yolo-person`, which uses YOLO person bounding boxes and takes the bottom center of the box as the player position.
In wide-angle tennis match footage, players are usually small, so object detection is generally more stable than pose estimation.

Pose estimation can use local ONNX files:

```text
weights/yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx
weights/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx
weights/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx
```

If local RTMPose / RTMO files are missing, `rtmlib` may try to download them into the user cache directory.

## 🚀 Usage

### First Run

1. Prepare an input video, and download weights from [GitHub Releases](https://github.com/yo-WASSUP/Good-Tennis/releases/latest) into `weights/`.
2. Run the basic command:

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png
```

3. The program will first try to automatically detect the four outer corners of the doubles tennis court.
4. If candidate court lines are detected, a preview window will be shown and `outputs/<video_name>/auto_court_preview.png` will be saved for inspection.
5. Press `Enter`/`Y` to accept the automatic result, or press `M`/`R`/`Esc` to switch to manual four-corner annotation.
6. During manual annotation, click the four outer corners in order: top-left, top-right, bottom-right, bottom-left.
7. The annotation result is saved to `outputs/<video_name>/court_annotations.txt`. Future runs in the same output directory will reuse this file.
8. After analysis, check `outputs/<video_name>/detect_<video_name>.mp4`, `detections.jsonl`, and `position_visualizations/`.

If the video camera angle, crop, or template image changes, delete `court_annotations.txt` in the corresponding output directory and annotate the four points again.

### Player Detection Modes

Use YOLO person detection by default:

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-model weights/yolo26s.pt
```

Enable Ultralytics built-in multi-object tracking to reduce cross-frame player box jumps:

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-tracker botsort
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-tracker bytetrack
```

The tracker `track_id` is only a weak continuity signal for player boxes; player identity is still maintained by `upper/lower` court region and court-coordinate continuity.

Switch to pose estimation:

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --player-detector pose --pose-family rtmpose
```

Use Ultralytics YOLO Pose:

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --player-detector pose --pose-family yolo-pose --yolo-pose-model weights/yolo11s-pose.pt
```

### Rally Detection

The program uses the court template image to determine whether the current frame is a match view, and automatically maintains rally state:

- A new rally starts after multiple consecutive frames match the court view.
- The current rally ends after multiple consecutive frames no longer match the court view.
- Rally IDs are written to `detections.jsonl` and shown in the output video stats overlay.
- Movement distance, speed, and other per-rally statistics reset at the start of each rally; match-level statistics continue accumulating.
- This logic depends on the template image and four-point court annotation. If the template is inaccurate, rally segmentation will also be inaccurate.

### Common Arguments

```text
--video-path                    Input video path, default videos/game9_Clip3.mp4
--output-dir                    Output directory, default outputs/<video_name>
--ball-model                    YOLO tennis ball detection model path, default weights/tennis-ball.pt
--ball-detector                 Ball detection backend: yolo, tracknet, or wasb, default yolo
--tracknet-model                TrackNet ball detection model path, default weights/tracknet_ball.pt (used with --ball-detector tracknet)
--wasb-model                    WASB-SBDT ball detection model path, default weights/wasb_tennis.pth (used with --ball-detector wasb)
--court-calibration             Camera calibration mode: keypoints (full calibration, enables speed/spin) or homography (homography-only, line calling only), default keypoints
--keypoint-model                Court keypoint detection model path, default weights/court_keypoints.pt
--shot-metrics true|false       Compute shot speed/spin, default true
--line-call singles|doubles|off Bounce line-calling court mode, off disables it, default doubles
--pose-family                   Pose model family: rtmpose, rtmo, or yolo-pose
--pose-mode                     RTMPose / RTMO mode: lightweight, balanced, performance
--yolo-pose-model               YOLO pose model path or model name, default weights/yolo11s-pose.pt
--player-detector               Player detector: yolo-person or pose, default yolo-person
--person-model                  YOLO person detection model path or model name, default weights/yolo26s.pt
--person-tracker                YOLO person box tracker: none, botsort, bytetrack, default botsort
--player-detect-interval        Player detection interval in frames; 1 detects every frame, use 2 or 3 for real-time preview
--template-path                 Court template image path; opens a file picker if omitted
--court-detection               Court corner detection mode: manual, auto, auto-fallback, default auto-fallback
--pose-roi true|false           Show pose detection ROI box, default true
--display true|false            Show OpenCV preview window, default true
--skeletons true|false          Show human skeletons, default true
--player-trajectories true|false Show player trajectories, default true
--court-trajectory true|false   Show court trajectory overlay, default true
--tennis-ball-trajectory true|false Show tennis ball trajectory, default true
--bounce-detection true|false   Detect and annotate tennis ball bounce points, default true
--mini-map true|false           Show court mini-map, default true
--player-stats true|false       Show player statistics, default true
--save-images                   Save processed frames
--performance-stats             Print performance timing
--visualize-positions true|false Generate heatmaps and scatter plots, default true
--audio true|false              Keep original video audio, default true
--language {zh,en}              Visualization language
```

## 📦 Outputs

Default output directory: `outputs/<video_name>/`

- `metadata.json`: Metadata for the video, models, court annotation, and output files; with `--shot-metrics true`, also includes `camera` (calibrated intrinsics/extrinsics, `recalibrated_at_frames` — the list of frames where drift triggered re-calibration).
- `detections.jsonl`: Frame-by-frame detection records, including rally ID, players, hands, court coordinates, speed, tennis ball coordinates, and post-processed bounce events.
- `bounce_events.json`: Bounce point list produced by full-trajectory post-processing, including frame index, image coordinates, confidence, and diagnostics.
- `cleaned_ball_trajectory.json`: Ball trajectory after filtering and short-gap interpolation; the final video uses this trajectory for drawing.
- `shot_metrics.json`: Generated when `--shot-metrics true`; per-shot metrics (`hit_frame`/`bounce_frame`/`hitter`/`speed_kmh`/`spin_coeff`/`spin_label`/`spin_confidence`/`line_call`/`fit_ok`/`rms_px`). Shots where the fit failed (`fit_ok=false`) stay in the list with `speed_kmh`/`spin_coeff` set to `null` rather than being silently dropped.
- `detect_<video_name>.mp4`: Output video with skeletons, trajectories, statistics, mini-map, and rally ID overlays.
- `court_annotations.txt`: Cached court annotation coordinates.
- `auto_court_preview.png`: Automatic court detection preview image, generated when automatic candidates are available.
- `position_visualizations/heatmaps/`: Player position heatmaps.
- `position_visualizations/scatter_plots/`: Player position scatter plots.

## 🎯 Shot Speed/Spin Accuracy and Validation

### Accuracy statement

- **Synthetic data (physics-simulated trajectories)**: against a real camera model plus a noisy physics-simulated trajectory, speed error is < 5% on 60fps footage and < 8% on 30fps footage (see the parametrized case `tests/test_trajectory3d.py::test_recovers_speed_within_tolerance_by_fps`, which quantifies how frame rate affects fit accuracy). Spin direction (topspin/slice) sign is always correct on synthetic data.
- **Real broadcast data**: acceptance criteria require a median error < 10%, but that requires ≥10 real broadcast serve clips with official speed captions (copyrighted/licensed material not included in this repo — the owner must supply it). **Running the real-data validation batch is currently an owner TODO.** The manifest format and command are documented below under "Validation tool".

### Known accuracy status

A diagnostic pass was run on the repo's own `videos/demo.mp4` (an amateur training-angle clip, not a broadcast camera position). **Every shot segment currently has `fit_ok=false`** (6 segments, `rms_px` ranging 14.17–73.05px, all above the fitter's `max_rms=12.0px` gate). Triaged in the brief's stated priority order (calibration → ball-detection dropout → fit residual):

1. **Calibration quality is good** — 14/14 keypoints, 4.82px reprojection error (well under the 15px degrade threshold). Calibration is ruled out.
2. **Ball-detection dropout is not the main driver** — whole-clip visibility is 93.5% (390 of 417 frames have a raw detection; the longest gap run is 12 frames, but it falls in the quiet period between shots, outside any shot segment). 3 of the 6 shot segments (`hit=48/230/386`) have **0% gap rate** within the segment, yet still produce `rms_px` of 14.17–25.35px, above threshold — meaning the fit residual stays high even with complete detection coverage, so missing frames alone don't explain it.
3. **The fit residual (`rms_px`) itself is broadly elevated** — median 22.33px across the 6 segments (min 14.17, max 73.05), all above threshold.

**Suggested direction (a diagnostic finding, not a fix already applied)**: inspecting `cleaned_ball_trajectory.json` shows every raw pixel coordinate in this clip is an **even integer** (all 390 raw detections have integer x/y with zero odd values) — consistent with the detector taking an argmax on a half-resolution heatmap and doubling the coordinate back up, which introduces systematic 2px quantization. Combined with this clip's amateur camera angle (not a fixed broadcast overhead position — see "Recording guidelines" below), that quantization noise is a plausible contributor to the elevated residuals. This clip also doesn't meet the recording guidelines below, which is itself a confound that can't be ruled out until real broadcast footage is validated. Fixing the fitter is out of scope for this task — this is a diagnosis for a follow-up task to act on.

### Validation tool

`tools/validate_speed.py` runs the full pipeline over every clip in a manifest, picks the `fit_ok=true` segment nearest each clip's `hit_frame_approx`, computes `|speed_kmh - caption_kmh| / caption_kmh`, and prints a table plus the median.

**Manifest format** (JSON array; fields documented in the file's module docstring):

```json
[
  {"video": "videos/serve_01.mp4", "hit_frame_approx": 142, "caption_kmh": 187},
  {"video": "videos/serve_02.mp4", "hit_frame_approx": 88, "caption_kmh": 201, "label": "Alcaraz ace"}
]
```

Put the real-clip manifest at `tools/serve_manifest.json` (ignored by `.gitignore` by default, never committed; the real broadcast clips themselves also shouldn't be added under `videos/` and committed).

```bash
# Run against real clips (once the owner has prepared tools/serve_manifest.json)
uv run tools/validate_speed.py --manifest tools/serve_manifest.json
uv run tools/validate_speed.py --manifest tools/serve_manifest.json --ball-detector tracknet

# Exercise the tool's own path with the repo's demo manifest (1 entry, demo.mp4,
# caption_kmh is an invented placeholder — this only verifies "parse manifest ->
# run pipeline -> select segment -> compute error -> print" works end to end,
# it is NOT an accuracy claim)
uv run tools/validate_speed.py --manifest tools/demo_manifest.json --ball-detector tracknet

# Run Step-3 diagnostics (gap rate + rms_px distribution) against an existing
# output directory without rerunning the pipeline
uv run tools/validate_speed.py --triage outputs/demo
```

Note: the first pipeline run for a given clip will trigger `main.py`'s own interactive court-confirmation window (`cv2.waitKey(0)` blocking, Enter/Y to accept or M/R/Esc to switch to manual corners) if that clip's output directory has no cached `court_annotations.txt` yet — this is existing `main.py` behavior, and the validation tool neither adds nor bypasses it. The tool keys output directories by video filename stem as `outputs/<video_stem>` (matching `main.py`'s own default output directory), so repeated validation runs on the same clip reuse the cached annotation instead of re-prompting.

### Recording guidelines

This tool doesn't tune the model — recording conditions set the accuracy ceiling. Before running real-data validation, please shoot footage that meets these requirements:

- **Camera position**: tripod fixed behind the baseline, centered, 2.5–3m high, angled down to cover the full court (calibration assumes a fixed camera — see "Camera calibration" above).
- **Resolution/frame rate**: 1080p, ≥30fps; **prefer 60fps** — it measurably improves both bounce-point and speed accuracy (synthetic-data error drops from < 8% at 30fps to < 5% at 60fps, see the quantified numbers under "Accuracy statement" above).
- **Avoid**: harsh midday shadows (ball/shadow confusion for the detector), and courts with heavily worn line paint (hurts court-keypoint detection and calibration quality).
- **Don't move the camera once recording starts** — moving it triggers Task 10's drift guard and a re-calibration warning (`metadata.json` records `recalibrated_at_frames`), and accuracy degrades for shot segments inside the re-calibration window.

## 🗂️ Project Structure

```text
main.py                    # CLI entry and argument parsing
pyproject.toml             # Project metadata and dependencies (managed by uv)
tools/
└── validate_speed.py      # Real-data speed validation tool (Task 11)
tennis_analysis/
├── system.py              # Main video analysis workflow: TennisAnalysisSystem
├── analysis/                # Shot segmentation, 3D trajectory fitting, spin classification, line calling
│   ├── segments.py            # hit->bounce shot segment extraction
│   ├── trajectory3d.py        # 3D trajectory fitting (speed/rms_px)
│   ├── physics.py             # Trajectory physics simulation (gravity+drag+Magnus, test-only)
│   ├── spin.py                 # Spin direction classification
│   ├── line_call.py            # Bounce IN/OUT line calling
│   ├── shot_metrics.py         # Per-segment metrics orchestration
│   └── shot_pipeline.py        # Full segments->metrics->spin->line_call chain
├── court/                   # Court annotation, coordinate mapping, and camera calibration
│   ├── camera.py               # CameraModel: projection / reprojection error
│   ├── camera_calibration.py   # Fixed-camera calibration + drift guard
│   └── keypoint_detector.py    # 14-point court keypoint detection
├── data/                  # JSON / JSONL outputs
├── detection/             # Tennis ball detection (yolo/tracknet/wasb), player detection, and pose detection
├── media/                 # Video and audio processing
├── tracking/              # Player, tennis ball trajectory, and rally tracking
└── visualization/         # Video overlays, statistics charts, and position plots
```

## 🙏 Acknowledgements

This project is built upon [yo-WASSUP/Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis) (Apache 2.0). Thanks to the original author for the open-source foundation.

Thanks to RTMPose, RTMO, and the OpenMMLab ecosystem for the pose estimation algorithm foundation, and to [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) for the lightweight pose estimation runtime.
Thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO object detection algorithm and toolchain.
Thanks to [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) for organizing and publishing the tennis dataset, which provides important reference material for tennis ball detection and trajectory analysis in this project.

Thanks to [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (MIT License) for the ball-detection backbone network and pretrained weights that power the `--ball-detector wasb` backend.

## License

This project is licensed under the Apache License 2.0. Third-party model weights are governed by their respective original licenses.
