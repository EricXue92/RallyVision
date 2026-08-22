# Model weights

Model weights are not committed to this repository.

Download the release assets from:

```text
https://github.com/yo-WASSUP/Good-Tennis/releases/latest
```

Place the downloaded files in this directory. The default paths used by the project are:

```text
weights/tennis-ball.pt
weights/yolo26s.pt
weights/yolo11s-pose.pt
weights/yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx
weights/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx
weights/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx
```

## Court keypoint detector (`weights/court_keypoints.pt`)

Used by `tennis_analysis/court/keypoint_detector.py::CourtKeypointDetector` (14-point
court keypoint heatmap network from
[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector),
Apache-2.0).

The pretrained weights are hosted on Google Drive (linked from that repo's README,
"Pretrained model" section):

```text
https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link
```

Download with `gdown` (no interactive auth needed, it's a public "anyone with the
link" file) and save it as `weights/court_keypoints.pt`:

```bash
uv run --with gdown python -c "
import gdown
gdown.download(id='1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG', output='weights/court_keypoints.pt', quiet=False)
"
```

Or manually download the file from the URL above via a browser and place it at
`weights/court_keypoints.pt`.

### Smoke test

Once the weights are in place, verify the detector end-to-end with
`tools/kp_smoke.py`: it runs `CourtKeypointDetector.detect` on the first frame
of `videos/demo.mp4` (override with `--video-path`), prints the valid point
count and a `CameraModel.calibrate` reprojection error, and saves a
numbered-points visualization to `outputs/kp_smoke.png`. If the weights file
is missing it prints a bilingual "缺权重 / weights missing" message and exits
0 instead of failing.

```bash
uv run python tools/kp_smoke.py
```

## TrackNet ball detector (`weights/tracknet_ball.pt`)

Used by `tennis_analysis/detection/tracknet_ball.py::TrackNetBallDetector`
(9-channel 3-frame-stack ball heatmap network from
[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet), Apache-2.0).
This is the `--ball-detector tracknet` backend (default remains `yolo`; see
`main.py --ball-detector`). It is a separate upstream repo/checkpoint from
the court keypoint detector above — same overall `BallTrackerNet` skeleton
but different `in_channels` (9 vs 3) and `out_channels` (256 vs 15), so the
network class is a separate copy in `tracknet_ball.py`, not a shared import.

The pretrained weights are hosted on Google Drive (linked from that repo's
README, "Pretrained model" section):

```text
https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing
```

Download with `gdown` (no interactive auth needed, it's a public "anyone with
the link" file) and save it as `weights/tracknet_ball.pt`:

```bash
uv run --with gdown python -c "
import gdown
gdown.download(id='1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl', output='weights/tracknet_ball.pt', quiet=False)
"
```

Or manually download the file from the URL above via a browser and place it
at `weights/tracknet_ball.pt`.
