# RallyVision: AI 网球鹰眼系统 🎾

<div align="center">

[![GitHub license](https://img.shields.io/github/license/EricXue92/RallyVision)](https://github.com/EricXue92/RallyVision/blob/main/LICENSE)

**基于计算机视觉的网球比赛视频分析工具**

本项目基于 [Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis)（Apache 2.0）发展而来

[中文](README.md) | [English](README_en.md)

</div>

### 🎬 视频分析结果

| RTMPose 姿态检测                                        | YOLO26s 人体检测                                        |
| ------------------------------------------------------- | ------------------------------------------------------- |
| ![RTMPose 姿态检测演示](assets/rtmpose_detect_demo.gif) | ![YOLO26s 人体检测演示](assets/yolo26s_detect_demo.gif) |

网球比赛远景里球员通常较小，目标检测一般比姿态估计更稳定。

## 📝 更新日志

- **2026-08-25**：弹跳检测默认改用 CatBoost 模型（移植自 [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject)，±2 帧滞后特征 + 0.45 阈值），落点召回大幅提升；权重放 `weights/ctb_regr_bounce.cbm` 自动启用，缺失时回退规则评分（下载方式见 `weights/README.md`）。
- **2026-08-23**：阶段三比赛层上线——击球类型分类（规则法）、回合切分、每分判定与自动计分（`--match-scoring`）、比赛统计（`match_stats.json`）、集锦导出（`--highlights`），以及比分修正重算工具 `tools/edit_point.py`（改判任意一分后整场从头重放自动重算）。
- **2026-08-22**：阶段二能力上线——固定机位相机标定（`--court-calibration`）、击球速度/旋转估计（`--shot-metrics`）、弹跳点 IN/OUT 判罚（`--line-call`）、新增 WASB-SBDT 球检测后端（`--ball-detector wasb`，与 yolo/tracknet 三选一），并提供真实数据球速验证工具 `tools/validate_speed.py`。
- **2026-07-02**：增加球员位置追踪和自动球场外角点检测
- **2026-06-22**：整理开源 README，增加网球弹跳点检测。
- **当前版本**：支持球员检测、网球检测、球场坐标映射、轨迹统计、回合检测、小地图、热力图/散点图、带标注视频输出，以及相机标定、击球速度/旋转估计、落点 IN/OUT 判罚与比赛层（击球分类/自动计分/统计/集锦/比分修正）。
- **迭代中**：弹跳检测已含双抛物线亚帧精化；当前重点是真实视频上的球速拟合精度（详见「击球速度/旋转精度与验证」章节的已知精度状态）。

## 🗺️ 路线图

- [x] 逐帧网球比赛视频分析
- [x] YOLO 人体检测与多种姿态模型可选
- [x] YOLO 网球检测集成
- [x] 手动/自动球场标注与球场坐标映射
- [x] 球员移动轨迹、速度、距离与回合统计
- [x] 网球轨迹与弹跳点标注
- [x] 标准网球场小地图叠加
- [x] 中/英文可视化文案
- [x] 热力图、散点图与检测数据导出
- [x] 固定机位相机标定（含漂移检测与自动重标定）
- [x] 基于 3D 弹道拟合的击球速度与旋转估计
- [x] 弹跳点 IN/OUT 判罚
- [x] 更多球检测后端（TrackNet、WASB-SBDT）
- [x] 真实数据验证工具（`tools/validate_speed.py`）
- [x] 比赛层：击球分类 / 回合切分 / 自动计分 / 统计 / 集锦（`--match-scoring`）
- [x] 比分修正重算工具（`tools/edit_point.py`）
- [ ] 更稳定的网球弹跳点识别
- [ ] 更准确的网球检测模型

---

## ✨ 功能

- **球员检测** - 默认使用 YOLO 人体框检测，也可切换到 RTMPose、RTMO 或 Ultralytics YOLO Pose 姿态估计。
- **网球检测** - 三种检测后端可选（`--ball-detector yolo|tracknet|wasb`）：YOLO 单帧检测框、TrackNet 多帧热力图、WASB-SBDT 多帧热力图；原始检测写入数据文件，最终视频绘制后处理过滤/插值后的干净轨迹。
- **球场标注** - 默认尝试自动检测双打外角点，失败后切换为手动点击四个外角点。
- **球场坐标映射** - 将图像坐标映射到标准双打网球场坐标，球场尺寸按 `10.97m x 23.77m` 建模。
- **相机标定** - `--court-calibration keypoints`（默认）用 14 点球场关键点检测对固定机位视频做完整相机标定（内参+外参），支持击球速度/旋转估计；`--court-calibration homography` 降级为仅单应性映射（无 speed/spin，只有落点判罚）。拍摄中途相机被移动时会自动检测漂移并重新标定（`metadata.json` 记录 `recalibrated_at_frames`）。
- **击球速度与旋转** - `--shot-metrics true`（默认）基于相机标定 + 物理弹道拟合，计算每拍的球速（km/h）和旋转方向（上旋/下旋/平击），拟合失败的段保留 `fit_ok=false` 并置空数值，不静默丢弃。
- **落点判罚** - `--line-call singles|doubles|off` 对每个弹跳点做单/双打边线 IN/OUT 判罚，含临界值容差带。
- **比赛层分析** - `--match-scoring true` 在逐拍指标之上跑完整比赛层：击球类型分类（发球/高压/截击/正手/反手）、回合切分、每分判定（含发球 fault 配对与双误）、自动计分（平分/占先、no-ad、6-6 抢七、best-of 3/5）与比赛统计。击球分类为规则法，正反手判定依赖姿态腕点，遮挡或腕点缺失时诚实记为 `unknown`，不硬猜。
- **比分修正重算** - `tools/edit_point.py` 可改判任意一分：计分状态机完整保存每分历史，改分后整场从头重放，局分/盘分/统计全部自动重算。
- **集锦导出** - `--highlights true` 把长回合与制胜分回合剪成带记分板叠加的集锦视频（需系统安装 ffmpeg，缺失时警告跳过不中断）。
- **球员位置追踪** - 记录球员球场坐标、移动轨迹、速度和距离。
- **回合检测** - 根据连续球场视图自动判断回合开始和结束，并在视频叠加层和检测数据中记录回合编号。
- **弹跳点检测** - 视频处理完成后，按整段网球轨迹做离群点清理、插值、速度计算，默认使用规则评分；干净球轨迹和弹跳点会在主画面和小地图上显示。
- **小地图叠加** - 在输出视频中显示标准网球场小地图，标注球员、网球和弹跳点位置。
- **位置图表** - 自动生成球员位置热力图和散点图。
- **中英文显示** - 可通过 `--language zh/en` 切换可视化文字。
- **本地运行** - 视频、模型和分析结果都保存在本地。

### 📊 球场与位置可视化

| 自动球场检测                     | 球员位置热力图                             | 球员位置散点图                             |
| -------------------------------- | ------------------------------------------ | ------------------------------------------ |
| ![自动球场检测](assets/auto.png) | ![球员位置热力图](assets/demo_heatmap.png) | ![球员位置散点图](assets/demo_scatter.png) |

## 🧩 系统要求

- [uv](https://docs.astral.sh/uv/)（自动管理 Python 版本和依赖）
- FFmpeg，并已加入系统 `PATH`
- OpenCV / PyTorch / Ultralytics / RTMLib / ONNX Runtime
- 推荐 NVIDIA GPU；CPU 可以运行，但视频分析速度会明显变慢

## ⚙️ 安装指南

本项目使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理依赖。安装 uv 后，在项目根目录执行：

```bash
uv sync
```

uv 会自动下载合适的 Python 版本、创建 `.venv` 并安装全部依赖（Windows / Linux / macOS 通用）。

### GPU 加速（Windows / NVIDIA）

默认依赖使用 CPU 版 PyTorch 和 ONNX Runtime。需要 GPU 加速时，先确认：

- 已安装 NVIDIA 显卡驱动，`nvidia-smi` 可以正常输出显卡信息。
- 推荐使用 CUDA 12.1 对应的 PyTorch wheel。
- 如果遇到 DLL 加载失败，先安装或修复 Microsoft Visual C++ Redistributable 2015-2022 x64。

PowerShell：

```bash
uv pip uninstall torch torchvision onnxruntime
uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
uv pip install onnxruntime-gpu==1.20.1
```

切回 CPU 版：

```bash
uv sync --reinstall
```

## 🧠 模型准备

首次运行前，请到项目的 GitHub Release 页面下载权重文件：

```text
https://github.com/yo-WASSUP/Good-Tennis/releases/latest
```

下载后把所有权重文件放到项目根目录下的 `weights/` 文件夹

如果缺少默认权重，程序会在启动时提示对应文件不存在。也可以通过 `--ball-model`、`--person-model`、`--yolo-pose-model` 指定自己的模型路径。

网球检测默认读取网球 YOLO 权重（`--ball-detector yolo`）。切换 `--ball-detector tracknet` 需要 `weights/tracknet_ball.pt`；切换 `--ball-detector wasb` 需要 `weights/wasb_tennis.pth`（来自 [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)，MIT License，下载方式见 `weights/README.md`）。

击球速度/旋转估计（`--shot-metrics`）与 `--court-calibration keypoints`（默认）需要 `weights/court_keypoints.pt` 做球场 14 点关键点检测；缺权重或标定失败（重投影误差 > 15px 或有效点 < 6）会打印双语警告并自动降级为仅单应性（`--court-calibration homography` 等价效果），此时只有落点判罚，没有球速/旋转。

球员检测模型由 `--player-detector` 切换。默认 `yolo-person`，使用 YOLO 人体框检测，并取检测框底部中点作为球员位置。

本地 RTMPose / RTMO 文件不存在时，`rtmlib` 可能会尝试在线下载到用户缓存目录。

## 🚀 使用指南

### 第一次运行流程

1. 准备输入视频，并从 [GitHub Releases](https://github.com/yo-WASSUP/Good-Tennis/releases/latest) 下载权重到 `weights/`。
2. 运行基础命令：

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png
```

3. 程序会先尝试自动检测网球双打场地四个外角点。
4. 检测到候选球场线时会显示预览窗口，并保存 `outputs/<视频文件名>/auto_court_preview.png` 供检查。
5. 按 `Enter`/`Y` 接受自动结果，按 `M`/`R`/`Esc` 切换到手动四角标注。
6. 手动标注时，按顺序点击左上、右上、右下、左下四个外角点。
7. 标注结果会保存到 `outputs/<视频文件名>/court_annotations.txt`。同一个输出目录下再次运行会复用这个文件。
8. 分析结束后，查看 `outputs/<视频文件名>/detect_<视频文件名>.mp4`、`detections.jsonl` 和 `position_visualizations/`。

如果换了视频视角、裁切方式或模板图，需要删除对应输出目录里的 `court_annotations.txt`，重新标注四点。

### 球员检测方式

默认使用 YOLO 人体框检测：

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-model weights/yolo26s.pt
```

启用 Ultralytics 内置多目标跟踪，减少球员框跨帧跳变：

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-tracker botsort
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --person-tracker bytetrack
```

跟踪器输出的 `track_id` 只作为球员框连续性的弱信号；球员身份仍以 `upper/lower` 半场和球场坐标连续性为准。

切换到姿态估计：

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --player-detector pose --pose-family rtmpose
```

使用 Ultralytics YOLO Pose：

```bash
uv run main.py --video-path videos/demo.mp4 --template-path templates/demo.png --player-detector pose --pose-family yolo-pose --yolo-pose-model weights/yolo11s-pose.pt
```

### 回合检测说明

程序会用球场模板图做比赛视图判断，并自动维护回合状态：

- 连续多帧匹配到球场视图时，判定新回合开始。
- 连续多帧没有匹配到球场视图时，判定当前回合结束。
- 回合编号会写入 `detections.jsonl`，并显示在输出视频的统计叠加层中。
- 每个回合开始时会重置该回合内的移动距离、速度等统计，整场统计继续累计。
- 这个逻辑依赖模板图和四点球场标注；如果模板图选得不准，回合切分也会不准。

### 常用参数

```text
--video-path                    输入视频路径，默认 videos/game9_Clip3.mp4
--output-dir                    输出目录，默认 outputs/<视频文件名>
--ball-model                    YOLO 网球检测模型路径，默认 weights/tennis-ball.pt
--ball-detector                 球检测后端：yolo、tracknet 或 wasb，默认 yolo
--tracknet-model                TrackNet 网球检测模型路径，默认 weights/tracknet_ball.pt（--ball-detector tracknet 时使用）
--wasb-model                    WASB-SBDT 网球检测模型路径，默认 weights/wasb_tennis.pth（--ball-detector wasb 时使用）
--court-calibration             相机标定方式：keypoints（完整相机标定，支持球速/旋转）或 homography（仅单应性，只有落点判罚），默认 keypoints
--keypoint-model                球场关键点检测模型路径，默认 weights/court_keypoints.pt
--shot-metrics true|false       是否计算击球速度/旋转，默认 true
--line-call singles|doubles|off 弹跳落点判罚场地模式，off 关闭，默认 doubles
--pose-family                   姿态模型族：rtmpose、rtmo 或 yolo-pose
--pose-mode                     RTMPose / RTMO 档位：lightweight、balanced、performance
--yolo-pose-model               YOLO pose 模型路径或模型名，默认 weights/yolo11s-pose.pt
--player-detector               球员检测方式：yolo-person 或 pose，默认 yolo-person
--person-model                  YOLO 人体检测模型路径或模型名，默认 weights/yolo26s.pt
--person-tracker                YOLO 人体框跟踪器：none、botsort、bytetrack，默认 botsort
--player-detect-interval        球员检测间隔帧数，1 表示每帧检测；实时预览可设为 2 或 3
--template-path                 球场模板图路径；不传时会弹出文件选择框
--court-detection               球场角点检测方式：manual、auto、auto-fallback，默认 auto-fallback
--pose-roi true|false           是否显示姿态检测 ROI 框，默认 true
--display true|false            是否显示 OpenCV 预览窗口，默认 true
--skeletons true|false          是否显示人体骨架，默认 true
--player-trajectories true|false 是否显示球员轨迹，默认 true
--court-trajectory true|false   是否显示球场轨迹叠加层，默认 true
--tennis-ball-trajectory true|false 是否显示网球轨迹，默认 true
--bounce-detection true|false   是否检测并标注网球弹跳点，默认 true
--mini-map true|false           是否显示球场小地图，默认 true
--player-stats true|false       是否显示球员统计信息，默认 true
--save-images                   保存处理后的每帧图像
--performance-stats             打印性能耗时
--visualize-positions true|false 是否生成热力图和散点图，默认 true
--audio true|false              是否保留原视频音频，默认 true
--language {zh,en}              选择界面语言
--match-scoring true|false      是否跑比赛层分析（击球类型/回合/每分判定/计分/统计），默认 false
--first-server upper|lower      首局发球方，默认 lower（离相机近的一方）
--upper-hand right|left         upper 方持拍手，默认 right
--lower-hand right|left         lower 方持拍手，默认 right
--best-of 3|5                   几盘制，默认 3
--no-ad true|false              是否用 no-ad（平分后金球）计分，默认 false
--highlights true|false         是否导出集锦视频（需 --match-scoring true 且系统装有 ffmpeg），默认 false
```

### 比赛层与比分修正

```bash
# 跑比赛层：击球分类 → 回合切分 → 每分判定 → 自动计分 → 统计（可选集锦）
uv run main.py --video-path videos/demo.mp4 --match-scoring true --first-server lower --highlights true

# 列出每一分（序号 / 帧区间 / 赢家 / 原因 / 该分开始前的比分）
uv run tools/edit_point.py --output-dir outputs/demo --list

# 改判第 12 分给 upper：状态机从头重放，match_score.json 与 match_stats.json 自动重算
uv run tools/edit_point.py --output-dir outputs/demo --point 12 --winner upper
```

自动判定不可能全对（视频截断、临界判罚等会产生 `reason=unknown` 的分）——这些分不会被静默丢弃，照样计入比分；判错的分用修正工具人工改判，改一分即整场自动重算。

## 📦 输出结果

默认输出到 `outputs/<视频文件名>/`：

- `metadata.json`：视频、模型、球场标注和输出文件元数据；`--shot-metrics true` 时额外含 `camera`（标定内参/外参、`recalibrated_at_frames` 漂移重标定帧列表）。
- `detections.jsonl`：逐帧检测记录，包含回合编号、球员、手部、球场坐标、速度、网球坐标和后处理弹跳点事件。
- `bounce_events.json`：整段轨迹后处理得到的弹跳点列表，包含帧号、图像坐标、置信度和诊断信息。
- `cleaned_ball_trajectory.json`：过滤和短缺失插值后的球轨迹，最终视频使用这份轨迹绘制。
- `shot_metrics.json`：`--shot-metrics true` 时生成，逐拍击球指标（`hit_frame`/`bounce_frame`/`hitter`/`speed_kmh`/`spin_coeff`/`spin_label`/`spin_confidence`/`line_call`/`fit_ok`/`rms_px`）；拟合失败的拍（`fit_ok=false`）保留在列表里，`speed_kmh`/`spin_coeff` 置 `null`，不静默丢弃；`--match-scoring true` 时每拍追加 `shot_type` 字段（serve/overhead/volley/forehand/backhand/unknown）。
- `match_score.json`：`--match-scoring true` 时生成——每分历史（`history`）、逐分比分快照（`score_timeline`）、每分判定明细（`points`：赢家/原因/帧区间/一发二发）与终局比分（`final`）；`tools/edit_point.py` 改判后原地重写。
- `match_stats.json`：`--match-scoring true` 时生成——六类击球计数与所在分胜率、发球统计（一发成功率/双误/均速/极速）、回合长度直方图、弹跳热图。
- `highlights.mp4`：`--highlights true` 时生成的集锦视频（长回合 + 制胜分回合，带记分板叠加）。
- `detect_<视频文件名>.mp4`：带骨架、轨迹、统计信息、小地图和回合编号叠加层的输出视频。
- `court_annotations.txt`：球场标注坐标缓存。
- `auto_court_preview.png`：自动球场检测预览图，触发自动检测候选时生成。
- `position_visualizations/heatmaps/`：球员位置热力图。
- `position_visualizations/scatter_plots/`：球员位置散点图。

## 🤖 Worker 模式

面向生产环境的另一种运行方式：不在本机手动跑 `main.py`，而是作为长驻/单次进程，向 TennisMatch backend 轮询领取分析任务、本机跑 pipeline、把 `report.json` 契约（`tools/report_builder.py` 聚合）和集锦视频回传给 backend。当前仅支持单打（`--line-call singles` 恒写死，见 `tools/worker.py`）。

### 环境变量

- `RV_BACKEND_BASE`：backend 地址，默认 `https://api.letstennis.app`。
- `RV_WORKER_TOKEN`：worker 鉴权 token（随每个请求带 `X-Worker-Token` 头），**必填**——缺失直接报错退出（exit code 2）。
- `RV_WORK_DIR`：本地工作目录根路径，默认 `~/rallyvision-jobs`。

### 用法

```bash
# 领一单跑完退出（内测手动模式）
RV_WORKER_TOKEN=xxx uv run tools/worker.py --once

# 常驻轮询：队列为空或一单处理完后，每 300 秒（默认，可用 --interval 调整）再领一次
RV_WORKER_TOKEN=xxx uv run tools/worker.py --loop --interval 300
```

### work_dir 结构

每个任务对应 `<RV_WORK_DIR>/<job_id>/`（默认 `~/rallyvision-jobs/<job_id>/`）：

- `input.mp4`：从 backend 下发的 `video_url` 下载的原始视频。
- `outputs/`：pipeline 完整输出目录，等价于本地跑 `main.py --output-dir` 的产物——`metadata.json`/`match_score.json`/`match_stats.json`/`shot_metrics.json`/`detections.jsonl`/`highlights.mp4` 等（字段说明见上方「输出结果」）。

视频与输出**不会被自动清理**，内测期本身就是排障素材。

### 失败排查

一单失败（下载失败、pipeline 子进程非零退出、`report_builder` 聚合失败等）时，worker 会把异常归类成 `error_code`（`court_not_detected` / `pipeline_error`）回传给 backend 的 `/fail` 端点，同时**保留本地现场**——不清理 `<work_dir>/<job_id>/` 下的 `input.mp4` 和 `outputs/`。排障时直接进 `~/rallyvision-jobs/<job_id>/` 检查有没有生成 `metadata.json`/`match_score.json` 等中间产物，或用相同参数（参考 `tools/worker.py::build_cli_args` 拼出的 argv）手动重跑 `main.py` 复现。若连失败上报本身也失败（网络问题等），只打日志不重试——任务会在 backend 侧 24 小时后自愈重置回队列。

## 🎯 击球速度/旋转精度与验证

### 精度声明

- **合成数据（物理仿真弹道）**：真实相机模型 + 物理弹道仿真（含噪声）下，60fps 素材球速误差 < 5%，30fps 素材误差 < 8%（见 `tests/test_trajectory3d.py::test_recovers_speed_within_tolerance_by_fps` 的参数化用例，量化了帧率对拟合精度的影响）。合成数据两类用例（上旋/切削）方向判定均正确（`test_recovers_topspin_sign`/`test_recovers_slice_sign`）。
- **真实转播数据**：验收标准要求中位数误差 < 10%，但需要 ≥10 条带官方球速字幕的真实转播发球片段（版权/授权素材，仓库内不提供，需 owner 手工准备）。**真实数据验证跑批目前是 owner 的 TODO**，manifest 格式和执行命令见下方「验证工具」。跑这批真实素材时，建议顺带做一次落点判罚（IN/OUT）人工抽查——挑约 20 个弹跳点核对 `line_call` 结果，压线 ±15cm 内允许判 "close"。

### 已知精度状态

用仓库自带的 `videos/demo.mp4`（业余训练视角，非转播机位）跑过一轮诊断，**目前全部击球段 `fit_ok=false`**（6 段 `rms_px` 范围 14.17–73.05px，均超过拟合器 `max_rms=12.0px` 门槛），已排查如下（顺序按 brief 的排查优先级，标定 → 缺帧 → 拟合残差）：

1. **相机标定质量良好** — 14/14 关键点，重投影误差 4.82px（远优于 15px 降级门槛），标定层已排除。
2. **球检测缺帧率不是主因** — 整段轨迹球检测可见率 93.5%（417 帧中 390 帧有原始检测，最长缺帧游程 12 帧，但落在两次击球之间的静默期，不在任何击球段内）；6 个击球段里有 3 段（`hit=48/230/386`）**缺帧率为 0%**，却仍然 `rms_px` 达 14.17–25.35px，超过门槛——说明即使球检测完整，拟合残差依然偏高，缺帧不是唯一/主要瓶颈。
3. **拟合残差（`rms_px`）本身普遍偏高** — 6 段中位数 22.33px（min 14.17，max 73.05），全部超阈值。

**建议方向（诊断结论，非已修复项）**：抽查 `cleaned_ball_trajectory.json` 发现该视频的原始像素坐标**全部是偶数**（390 个原始检测点，x/y 均为整数且无一奇数），指向检测器在半分辨率热力图上取 argmax 再乘 2 还原坐标，造成系统性 2px 量化——这类量化噪声叠加上业余机位（`videos/demo.mp4` 非固定转播俯角、镜头稍有畸变）的透视条件，可能是残差普遍偏高的可信解释之一；此外该素材本身不满足下方「拍摄规范」（非底线正后方固定机位），也是真实转播素材验证前不能排除的变量。本任务范围不含修复拟合器，仅诊断记录，供后续任务参考。

### 验证工具

`tools/validate_speed.py` 对 manifest 里每条素材跑一遍完整 pipeline，取离 `hit_frame_approx` 最近的 `fit_ok=true` 段，算 `|speed_kmh - caption_kmh| / caption_kmh` 相对误差，打印表格与中位数。

**Manifest 格式**（JSON 数组，字段见文件头注释）：

```json
[
  {
    "video": "videos/serve_01.mp4",
    "hit_frame_approx": 142,
    "caption_kmh": 187
  },
  {
    "video": "videos/serve_02.mp4",
    "hit_frame_approx": 88,
    "caption_kmh": 201,
    "label": "Alcaraz ace"
  }
]
```

真实素材 manifest 放到 `tools/serve_manifest.json`（`.gitignore` 默认忽略，不入库；真实转播片段本身也不建议放进 `videos/` 提交到 git）。

```bash
# 用真实素材跑（owner 准备好 tools/serve_manifest.json 后执行）
uv run tools/validate_speed.py --manifest tools/serve_manifest.json
uv run tools/validate_speed.py --manifest tools/serve_manifest.json --ball-detector tracknet

# 用仓库自带的演示 manifest 验证工具链路本身（1 条 demo.mp4，caption_kmh 为臆造值，
# 只验证「manifest 解析 → 跑 pipeline → 选段 → 算误差 → 打印」链路能跑通，不是精度声明）
uv run tools/validate_speed.py --manifest tools/demo_manifest.json --ball-detector tracknet

# 对已有输出目录单独跑 Step3 诊断（缺帧率 + rms_px 分布），不重跑 pipeline
uv run tools/validate_speed.py --triage outputs/demo
```

注意：每条素材第一次跑 pipeline 时，若对应输出目录没有缓存的 `court_annotations.txt`，会触发 `main.py` 本身的交互式球场确认窗口（`cv2.waitKey(0)` 阻塞，需要按 Enter/Y 或 M/R/Esc）——这是 pipeline 一贯行为，验证工具不新增/不绕过这个交互；工具按视频文件名把输出目录固定为 `outputs/<video_stem>`（与 `main.py` 自己的默认输出目录一致），同一支视频重复验证会复用已缓存的标注，不会重复弹窗。

### 拍摄规范

不调模型，拍摄条件即精度上限。真实数据验证前请按以下要求拍摄：

- **机位**：三脚架固定在底线后中央、高 2.5–3m，俯角覆盖整个球场（标定假设固定机位，见上方「相机标定」）。
- **分辨率/帧率**：1080p，≥30fps；**优先 60fps**——落点与球速精度提升明显（合成数据下误差从 30fps 的 < 8% 降到 60fps 的 < 5%，见上方「精度声明」的量化数据）。
- **避免**：正午强阴影（球体/阴影混淆检测器）、球线严重磨损的场地（影响球场关键点检测和标定质量）。
- **开拍后相机不能动** — 移动机位会触发 Task 10 的漂移守卫重标定警告（`metadata.json` 记录 `recalibrated_at_frames`），且重标定窗口内的击球段精度会下降。

## 🗂️ 项目结构

```text
main.py                    # 命令行入口和参数解析
pyproject.toml             # 项目元数据与依赖（uv 管理）
tools/
└── validate_speed.py      # 真实数据球速验证工具（Task 11）
tennis_analysis/
├── system.py              # 视频分析主流程 TennisAnalysisSystem
├── analysis/               # 击球段提取、3D 弹道拟合、旋转分类、落点判罚
│   ├── segments.py          # hit->bounce 击球段提取
│   ├── trajectory3d.py      # 3D 弹道拟合（速度/rms_px）
│   ├── physics.py           # 弹道物理仿真（重力+阻力+Magnus，测试用）
│   ├── spin.py               # 旋转方向分类
│   ├── line_call.py          # 落点 IN/OUT 判罚
│   ├── shot_metrics.py       # 单段指标编排
│   └── shot_pipeline.py      # segments->metrics->spin->line_call 全链路
├── court/                  # 球场标注、坐标映射与相机标定
│   ├── camera.py             # CameraModel：投影/重投影误差
│   ├── camera_calibration.py # 固定机位标定 + 漂移守卫
│   └── keypoint_detector.py  # 球场 14 关键点检测
├── data/                  # JSON / JSONL 输出
├── detection/             # 网球检测（yolo/tracknet/wasb）、球员检测和姿态检测
├── media/                 # 视频音频处理
├── tracking/              # 球员、网球轨迹和回合追踪
└── visualization/         # 视频叠加层、统计图和位置图
```

## 🙏 致谢

本项目基于 [yo-WASSUP/Good-Tennis](https://github.com/yo-WASSUP/Good-Tennis)（Apache 2.0）发展而来，感谢原作者的开源工作。

感谢 RTMPose、RTMO 和 OpenMMLab 生态提供的姿态估计算法基础，以及 [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) 提供的轻量姿态估计运行库。

感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的 YOLO 目标检测算法与工具链。

感谢 [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) 项目整理并公开网球数据集，为本项目的网球检测与轨迹分析提供了重要参考。

感谢 [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)（MIT License）提供的网球检测骨干网络与预训练权重，为 `--ball-detector wasb` 后端提供了重要基础。

## 许可证

本项目代码使用 Apache License 2.0。随项目使用的第三方模型权重许可证以其实际来源为准。
