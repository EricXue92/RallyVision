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
court keypoint heatmap network, architecture per the TrackNet paper (Huang et al.
2019); reference implementation:
[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector).
That repo declares no license (GitHub API `license` field is `None`, no LICENSE
file in the repo root) — usage/redistribution status is unverified; do not treat
it as Apache-2.0 or any other known license.

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
(9-channel 3-frame-stack ball heatmap network, architecture per the TrackNet
paper (Huang et al. 2019); reference implementation:
[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)). That repo
declares no license (GitHub API `license` field is `None`, no LICENSE file in
the repo root) — usage/redistribution status is unverified; do not treat it
as Apache-2.0 or any other known license.
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

## WASB ball detector (`weights/wasb_tennis.pth`)

Used by `tennis_analysis/detection/wasb_ball.py::WASBBallDetector`
(9-channel 3-frame-stack HRNet heatmap network, one heatmap channel per
frame in the window — see below). This is the `--ball-detector wasb`
backend (default remains `yolo`; see `main.py --ball-detector`).

Upstream: [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) ("Widely
Applicable Strong Baseline for Sports Ball Detection and Tracking",
arXiv:2311.05237). **That repo's root `LICENSE.md` is MIT License**
(Copyright (c) 2023 NTT Communications Corporation) — verified by reading
the file directly, unlike the yastrebksv-family repos above (TrackNet, court
keypoint detector) which declare no license at all. The vendored network
class (`_build_hrnet_class()` in `wasb_ball.py`) is a line-by-line copy of
the repo's `src/models/hrnet.py::HRNet`, itself adapted from
[HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
`cls_hrnet.py` (also MIT License, authored by Bin Xiao / Bowen Cheng — see
that file's header). The only structural change from the upstream copy is
switching one config-access site from attribute access (`cfg.MODEL.EXTRA`,
which only works with hydra/OmegaConf `DictConfig` objects) to plain dict
indexing, since this project doesn't depend on hydra/omegaconf; see the
comment at that call site in `wasb_ball.py`.

The pretrained tennis weights are hosted on Google Drive (`MODEL_ZOO.md`,
"WASB (Ours)" row × "Tennis" column):

```text
https://drive.google.com/file/d/14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z/view?usp=drive_link
```

Download with `gdown` and save it as `weights/wasb_tennis.pth`:

```bash
uv run --with gdown python -c "
import gdown
gdown.download(id='14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z', output='weights/wasb_tennis.pth', quiet=False)
"
```

Or manually download the file from the URL above via a browser and place it
at `weights/wasb_tennis.pth`. If the file is missing when `--ball-detector
wasb` is selected, `TennisAnalysisSystem` prints a bilingual warning and
falls back to the `yolo` backend rather than crashing the whole run.

Loading has been verified against the real downloaded checkpoint: vendored
`HRNet(_HRNET_CFG)` + `model.load_state_dict(state_dict, strict=True)`
produces zero missing/unexpected keys, and a forward pass on a
`(1, 9, 288, 512)` input produces the expected `(1, 3, 288, 512)` output.

### Backend comparison (honest numbers, `videos/demo.mp4`, first 300 frames, no ROI)

| Backend  | Visible frames  | Runtime (300 frames) | Throughput |
| -------- | --------------- | --------------------- | ---------- |
| yolo     | 292/300 (97.3%) | 8.60s  | 34.9 fps |
| tracknet | 289/300 (96.3%) | 34.96s | 8.6 fps  |
| wasb     | 289/300 (96.3%) | 15.56s | 19.3 fps |

Measured on this demo video (a broadcast source), neither heatmap backend
beats plain YOLO on visible-frame rate — same finding as the earlier
TrackNet-only comparison (96.3% vs 97.3%). WASB matches TrackNet's
visible-frame rate exactly on this clip while running roughly 2.2x faster
(smaller 512x288 single-scale HRNet branch vs TrackNet's 640x360 U-Net-style
network), both measured on Apple Silicon MPS. This is not a controlled
precision/recall study (no ground-truth ball positions, just "did *some*
candidate clear the confidence threshold"), so treat these as rough
backend-selection signals, not accuracy claims. The default backend stays
`yolo`; `tracknet`/`wasb` remain opt-in experiments via `--ball-detector`.

## Bounce detection model (`weights/ctb_regr_bounce.cbm`)

Used by `tennis_analysis/analysis/bounce.py::BounceDetector` as the default
bounce (landing point) detector: a CatBoost regressor over ±2-frame lagged
x/y image-trajectory features, ported from
[yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject)
(`bounce_detector.py`, threshold 0.45, consecutive-frame merge). That repo
declares no license (no LICENSE file in the repo root) — usage/redistribution
status is unverified, same caveat as the court keypoint detector above.

The pretrained model is hosted on Google Drive (linked from that repo's
README, "Bounce detection" section):

```text
https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view?usp=drive_link
```

Download with `gdown` and save it as `weights/ctb_regr_bounce.cbm`:

```bash
uv run --with gdown python -c "import gdown; gdown.download(id='1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ', output='weights/ctb_regr_bounce.cbm', quiet=False)"
```

When the file is present it is picked up automatically (no CLI flag needed);
when missing, `BounceDetector` falls back to the legacy rule-based scoring
chain. An explicit `--bounce-classifier` path still overrides the default.
