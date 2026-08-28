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

## TrackNetV4 ball detector (`weights/tracknet_v4_*.pt`)

Upstream: [TrackNetV4/TrackNetV4](https://github.com/TrackNetV4/TrackNetV4),
**MIT licensed** — unlike the TrackNet / court-keypoint / bounce checkpoints
above, redistribution rights here are explicit. The model is a TrackNetV2
encoder-decoder plus a "motion prompt" attention layer: adjacent-frame
grayscale differences are power-normalized into attention maps that modulate
the per-frame output heatmaps, suppressing static-background responses.

Upstream ships Keras 3 `.keras` checkpoints. To keep TensorFlow out of the
worker runtime they are converted **offline** to PyTorch — a `.keras` file is
just a zip whose `model.weights.h5` holds plain arrays, so h5py alone suffices:

```bash
uv run --with h5py tools/convert_tracknetv4.py \
    --keras ~/Desktop/new_tennis/best_model_V1_NF_RIO_10u_e17.keras \
    --out weights/tracknet_v4_typeA.pt
```

The fusion type (Type A / Type B / no-fusion V2 baseline) is auto-detected
from the custom layer class names in the checkpoint's `config.json`. Use it
with `--ball-detector tracknetv4 --tracknetv4-model <path> --tracknetv4-fusion
<A|B|none>`; the fusion flag must match the weights.

The port was verified numerically against the upstream Keras graph running on
Keras 3's torch backend (custom layers reimplemented with `keras.ops`, no
TensorFlow): on a real three-frame window from `videos/demo.mp4`, all three
checkpoints agree to `max|diff| ~= 6e-6` with identical argmax peak positions.

Two things the port has to get exactly right, both silent-failure-shaped:

- **Input is RGB in forward temporal order** (`[t-2, t-1, t]`), 512x288, `/255`
  — not the BGR / newest-first stacking the `tracknet` backend uses. The
  9-channel reorder happens in `_TorchTrackNetV4Adapter`.
- **`BatchNormalization` normalizes over W, not C.** Upstream writes
  `BatchNormalization()` (default `axis=-1`) over `channels_first` data, so the
  normalized axis is width; the saved parameter lengths (512/256/128/64 = each
  stage's width) confirm it. `_WidthBatchNorm` replicates this; `nn.BatchNorm2d`
  would be wrong. A side effect: **input width is locked to 512** — the BN
  parameters are width-sized, so the network cannot be run at another width.

### Backend comparison (honest numbers, no ROI / no gating)

Only `new_tennis`-trained checkpoints are available (upstream's `RESULT.md`
Download links are `#` placeholders, so the stronger standard-tennis-dataset
checkpoints are not published). Fill rate alone cannot separate true detections
from false ones, so trajectory plausibility is reported alongside it: implausible
frame-to-frame jumps (>150px at 720p, scaled by resolution like the player-box
gating does) and the max residual of a local quadratic fit over 5-frame
all-visible windows — a ball follows a near-parabola in image space, so a
well-tracked trajectory has a residual of a couple of pixels.

**Broadcast footage — the incumbent wins decisively:**

| Video | Backend | Fill rate | Jumps | Residual (median / p90) |
| ----- | ------- | --------- | ----- | ----------------------- |
| `demo.mp4` (Qatar Open) | tracknet @0.5 | 94.2% | 6/383 | 1.5 / 6.7 |
| `demo.mp4` | tracknetv4 A @0.3 | 99.8% | 125/414 | 49.3 / 174.1 |
| `demo.mp4` | tracknetv4 B @0.3 | 92.1% | 106/358 | 36.5 / 148.8 |
| job `7bb0934f` (US Open) | tracknet @0.5 | 94.7% | 4/325 | 1.2 / 2.9 |
| job `7bb0934f` | tracknetv4 A @0.3 | 93.3% | 50/320 | 17.0 / 109.1 |
| job `7bb0934f` | tracknetv4 B @0.3 | 77.2% | 37/241 | 11.7 / 101.2 |

**Amateur footage — the result flips.** Clips are the three `Amateur *.mp4`
samples from [VKorpelshoek/GridTrackNet](https://github.com/VKorpelshoek/GridTrackNet)
(1920x1080 club-court footage, far closer to what production actually ingests
than any broadcast clip available locally):

| Video | Backend | Fill rate | Jumps | Residual (median / p90) |
| ----- | ------- | --------- | ----- | ----------------------- |
| Amateur Hardcourt | tracknet @0.5 | 65.5% | 1/225 | 1.1 / 2.7 |
| Amateur Hardcourt | **tracknetv4 A @0.5** | **79.5%** | 3/293 | **0.7 / 2.4** |
| Amateur Grass | tracknet @0.5 | 59.8% | 5/87 | 1.1 / 124.2 |
| Amateur Grass | **tracknetv4 A @0.5** | 58.5% | **1/87** | **0.6 / 3.4** |
| Amateur Clay | tracknet @0.5 | 76.9% | 10/276 | 1.1 / 15.7 |
| Amateur Clay | tracknetv4 A @0.5 | 70.6% | **7/256** | 1.0 / 34.6 |

On Amateur Hardcourt TrackNetV4 Type A finds 85 frames the incumbent misses,
**at equal-or-better trajectory quality** — and a manual crop check of a
12-frame sample confirms 11 of them contain a visible ball, i.e. they are real
recoveries, not false positives. That is exactly the far-court / serve
detection hole that shows up downstream as missing bounce points.

Two operating-point notes:

- **Keep the confidence threshold at the 0.5 default.** Lowering it to 0.3
  inflates fill rate with junk on every clip (residual p90 jumps to 75-216px).
  The low-threshold rows above are shown to document that, not to recommend it.
- Type A beats Type B consistently; the no-fusion baseline is never best.
- TrackNetV4 also runs ~1.5x faster (12-13 fps vs 4.6-8.9 on MPS) despite the
  motion-attention branch, because 512x288 is a smaller input than 640x360.

Why it loses on broadcast and wins on amateur: the checkpoints are trained on
upstream's `new_tennis` set, and the incumbent `tracknet` weights are trained on
the standard broadcast tennis dataset — each wins in its own domain. The lower
input resolution (512x288 vs 640x360, and the width is not adjustable — see the
BN note above) is a real handicap for small far-court balls that the amateur
clips' 1080p source partly offsets.

The default backend is unchanged (`tracknet` in the worker); `tracknetv4` is
opt-in pending an A/B on actual user uploads.

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
