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

- **2026-08-22**：阶段二能力上线——固定机位相机标定（`--court-calibration`）、击球速度/旋转估计（`--shot-metrics`）、弹跳点 IN/OUT 判罚（`--line-call`）、新增 WASB-SBDT 球检测后端（`--ball-detector wasb`，与 yolo/tracknet 三选一），并提供真实数据球速验证工具 `tools/validate_speed.py`。
- **2026-07-02**：增加球员位置追踪和自动球场外角点检测
- **2026-06-22**：整理开源 README，增加网球弹跳点检测。
- **当前版本**：支持球员检测、网球检测、球场坐标映射、轨迹统计、回合检测、小地图、热力图/散点图和带标注视频输出。
- **实验功能**：网球弹跳点检测仍在迭代中，适合研究和二次开发使用。

---

## ✨ 功能

- **球员检测** - 默认使用 YOLO 人体框检测，也可切换到 RTMPose、RTMO 或 Ultralytics YOLO Pose 姿态估计。
- **网球检测** - 三种检测后端可选（`--ball-detector yolo|tracknet|wasb`）：YOLO 单帧检测框、TrackNet 多帧热力图、WASB-SBDT 多帧热力图；原始检测写入数据文件，最终视频绘制后处理过滤/插值后的干净轨迹。
- **球场标注** - 默认尝试自动检测双打外角点，失败后切换为手动点击四个外角点。
- **球场坐标映射** - 将图像坐标映射到标准双打网球场坐标，球场尺寸按 `10.97m x 23.77m` 建模。
- **相机标定** - `--court-calibration keypoints`（默认）用 14 点球场关键点检测对固定机位视频做完整相机标定（内参+外参），支持击球速度/旋转估计；`--court-calibration homography` 降级为仅单应性映射（无 speed/spin，只有落点判罚）。拍摄中途相机被移动时会自动检测漂移并重新标定（`metadata.json` 记录 `recalibrated_at_frames`）。
- **击球速度与旋转** - `--shot-metrics true`（默认）基于相机标定 + 物理弹道拟合，计算每拍的球速（km/h）和旋转方向（上旋/下旋/平击），拟合失败的段保留 `fit_ok=false` 并置空数值，不静默丢弃。
- **落点判罚** - `--line-call singles|doubles|off` 对每个弹跳点做单/双打边线 IN/OUT 判罚，含临界值容差带。
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
```

## 📦 输出结果

默认输出到 `outputs/<视频文件名>/`：

- `metadata.json`：视频、模型、球场标注和输出文件元数据；`--shot-metrics true` 时额外含 `camera`（标定内参/外参、`recalibrated_at_frames` 漂移重标定帧列表）。
- `detections.jsonl`：逐帧检测记录，包含回合编号、球员、手部、球场坐标、速度、网球坐标和后处理弹跳点事件。
- `bounce_events.json`：整段轨迹后处理得到的弹跳点列表，包含帧号、图像坐标、置信度和诊断信息。
- `cleaned_ball_trajectory.json`：过滤和短缺失插值后的球轨迹，最终视频使用这份轨迹绘制。
- `shot_metrics.json`：`--shot-metrics true` 时生成，逐拍击球指标（`hit_frame`/`bounce_frame`/`hitter`/`speed_kmh`/`spin_coeff`/`spin_label`/`spin_confidence`/`line_call`/`fit_ok`/`rms_px`）；拟合失败的拍（`fit_ok=false`）保留在列表里，`speed_kmh`/`spin_coeff` 置 `null`，不静默丢弃。
- `detect_<视频文件名>.mp4`：带骨架、轨迹、统计信息、小地图和回合编号叠加层的输出视频。
- `court_annotations.txt`：球场标注坐标缓存。
- `auto_court_preview.png`：自动球场检测预览图，触发自动检测候选时生成。
- `position_visualizations/heatmaps/`：球员位置热力图。
- `position_visualizations/scatter_plots/`：球员位置散点图。

## 🎯 击球速度/旋转精度与验证

### 精度声明

- **合成数据（物理仿真弹道）**：真实相机模型 + 物理弹道仿真（含噪声）下，60fps 素材球速误差 < 5%，30fps 素材误差 < 8%（见 `tests/test_trajectory3d.py::test_recovers_speed_within_tolerance_by_fps` 的参数化用例，量化了帧率对拟合精度的影响）。旋转方向（上旋/下旋）判定符号在合成数据下始终正确。
- **真实转播数据**：验收标准要求中位数误差 < 10%，但需要 ≥10 条带官方球速字幕的真实转播发球片段（版权/授权素材，仓库内不提供，需 owner 手工准备）。**真实数据验证跑批目前是 owner 的 TODO**，manifest 格式和执行命令见下方「验证工具」。

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
  {"video": "videos/serve_01.mp4", "hit_frame_approx": 142, "caption_kmh": 187},
  {"video": "videos/serve_02.mp4", "hit_frame_approx": 88, "caption_kmh": 201, "label": "Alcaraz ace"}
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
