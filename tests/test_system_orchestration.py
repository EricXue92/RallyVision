"""Task 10 review fix (Important finding): unit coverage for the system.py glue
code implementing R3 (`_resolve_player_positions`) and R12 (`_camera_for_frame`
span lookup + `_calibrate_camera`'s multi-span drift orchestration), plus
`_split_records_into_rallies`. Only the pure math helpers and
`compute_shot_metrics_entries` had tests before this file; the orchestration
around them (previously exercised only by one manual smoke run on a 417-frame
clip that structurally cannot hit the drift/multi-span/whole-video-fallback
branches) had zero automated coverage.

`_camera_for_frame`/`_resolve_player_positions`/`_split_records_into_rallies`
are exercised directly on a bare `TennisAnalysisSystem.__new__(...)` instance
(skips `__init__`, so no video file / YOLO / RTMPose weights needed — these
three methods only touch plain data + class-level constants).

`_calibrate_camera`'s multi-span drift orchestration needs `cv2.VideoCapture`
and `CourtKeypointDetector`; rather than mock those with cv2/torch test
doubles, we monkeypatch the module-level globals `system.py::load_runtime_dependencies`
normally installs (`cv2`, `CourtKeypointDetector`) with lightweight fakes, while
leaving `CameraModel`/`median_keypoints_over_frames`/`keypoints_drifted`/
`COURT_KEYPOINTS_M` wired to the REAL implementations — so the test exercises
real `cv2.solvePnP`-based calibration (via `tennis_analysis.court.camera`,
which imports its own `cv2` independent of the fake we install into
`system.py`'s globals) against synthetic-but-real court-keypoint projections,
not a fully mocked calibration.
"""
import cv2
import numpy as np

import tennis_analysis.system as system_module
from tennis_analysis.system import TennisAnalysisSystem
from tennis_analysis.court.camera import CameraModel
from tennis_analysis.court.camera_calibration import keypoints_drifted, median_keypoints_over_frames
from tennis_analysis.court.keypoint_detector import COURT_KEYPOINTS_M

W, H = 1280, 720


# ----------------------------------------------------------------------------
# _camera_for_frame (R12 span lookup) — pure function over plain data, no
# mocking needed at all.
# ----------------------------------------------------------------------------

def test_camera_for_frame_single_span_covers_whole_range():
    spans = [{"start_frame": 1, "end_frame": 100, "camera": "cam0"}]
    assert TennisAnalysisSystem._camera_for_frame(spans, 1) == "cam0"
    assert TennisAnalysisSystem._camera_for_frame(spans, 50) == "cam0"
    assert TennisAnalysisSystem._camera_for_frame(spans, 100) == "cam0"


def test_camera_for_frame_multi_span_picks_span_covering_frame():
    spans = [
        {"start_frame": 1, "end_frame": 900, "camera": "cam0"},
        {"start_frame": 901, "end_frame": 1300, "camera": "cam1"},
    ]
    assert TennisAnalysisSystem._camera_for_frame(spans, 500) == "cam0"
    assert TennisAnalysisSystem._camera_for_frame(spans, 1000) == "cam1"


def test_camera_for_frame_boundary_frames_go_to_the_span_that_owns_them():
    spans = [
        {"start_frame": 1, "end_frame": 900, "camera": "cam0"},
        {"start_frame": 901, "end_frame": 1300, "camera": "cam1"},
    ]
    assert TennisAnalysisSystem._camera_for_frame(spans, 900) == "cam0"
    assert TennisAnalysisSystem._camera_for_frame(spans, 901) == "cam1"


def test_camera_for_frame_beyond_last_span_falls_back_to_last_camera():
    spans = [
        {"start_frame": 1, "end_frame": 900, "camera": "cam0"},
        {"start_frame": 901, "end_frame": 1300, "camera": "cam1"},
    ]
    # 理论上不该发生（spans 覆盖到 total_frames 为止），但防御性兜底：
    # 越界帧号仍返回最后一段的相机，而不是 IndexError/None。
    assert TennisAnalysisSystem._camera_for_frame(spans, 99999) == "cam1"


def test_camera_for_frame_empty_spans_returns_none():
    assert TennisAnalysisSystem._camera_for_frame([], 5) is None


# ----------------------------------------------------------------------------
# _resolve_player_positions (R3 whole-video fallback)
# ----------------------------------------------------------------------------

def _bare_system():
    """跳过 __init__（需要视频文件 + YOLO/RTMPose 权重），这三个方法只碰
    普通数据结构 + 类级常量，不需要真实实例状态。"""
    return TennisAnalysisSystem.__new__(TennisAnalysisSystem)


def test_resolve_player_positions_uses_rally_position_when_present():
    obj = _bare_system()
    rally_positions = {"upper": [1.0, 2.0], "lower": [3.0, 4.0]}
    whole_video_positions = {"upper": [9.0, 9.0], "lower": [9.0, 9.0]}
    resolved, missing = obj._resolve_player_positions(rally_positions, whole_video_positions)
    assert resolved == {"upper": [1.0, 2.0], "lower": [3.0, 4.0]}
    assert missing == []


def test_resolve_player_positions_falls_back_to_whole_video_when_rally_missing():
    obj = _bare_system()
    rally_positions = {"upper": None, "lower": [3.0, 4.0]}
    whole_video_positions = {"upper": [9.0, 9.0], "lower": [1.0, 1.0]}
    resolved, missing = obj._resolve_player_positions(rally_positions, whole_video_positions)
    assert resolved["upper"] == [9.0, 9.0]  # 回合内缺 -> 退化到全视频中位数
    assert resolved["lower"] == [3.0, 4.0]  # 回合内有 -> 优先用回合内的
    assert missing == []


def test_resolve_player_positions_marks_missing_when_both_absent():
    obj = _bare_system()
    rally_positions = {"upper": None, "lower": None}
    whole_video_positions = {"upper": None, "lower": [2.0, 2.0]}
    resolved, missing = obj._resolve_player_positions(rally_positions, whole_video_positions)
    assert missing == ["upper"]
    assert resolved["upper"] == [5.485, 0.0]  # 占位坐标，仅防 extract_segments 崩，会被下游丢弃
    assert resolved["lower"] == [2.0, 2.0]


def test_resolve_player_positions_missing_sides_matches_downstream_discard_filter():
    """复刻 _run_shot_and_line_call_pipeline 里
    `entries = [e for e in entries if e["hitter"] not in missing_sides]` 的用法，
    验证 _resolve_player_positions 返回的 missing 列表能被这个惯用法正确消费
    （即真正把该侧的 shot_metrics 条目滤掉，而不仅仅是「返回了一个列表」）。"""
    obj = _bare_system()
    _, missing_sides = obj._resolve_player_positions(
        {"upper": None, "lower": [1.0, 1.0]}, {"upper": None, "lower": None}
    )
    entries = [{"hitter": "upper", "x": 1}, {"hitter": "lower", "x": 2}]
    filtered = [entry for entry in entries if entry["hitter"] not in missing_sides]
    assert filtered == [{"hitter": "lower", "x": 2}]


# ----------------------------------------------------------------------------
# _split_records_into_rallies
# ----------------------------------------------------------------------------

def test_split_records_into_rallies_splits_on_large_gap():
    obj = _bare_system()
    records = [{"frame": f} for f in (1, 2, 3)] + [{"frame": f} for f in (50, 51, 52)]
    rallies = obj._split_records_into_rallies(records)
    assert len(rallies) == 2
    assert (rallies[0]["start"], rallies[0]["end"]) == (1, 3)
    assert (rallies[1]["start"], rallies[1]["end"]) == (50, 52)


def test_split_records_into_rallies_keeps_small_gap_in_one_rally():
    obj = _bare_system()
    # gap 10-3=7 <= RALLY_GAP_FRAMES(15) -> 同一回合
    records = [{"frame": f} for f in (1, 2, 3, 10, 11)]
    rallies = obj._split_records_into_rallies(records)
    assert len(rallies) == 1
    assert (rallies[0]["start"], rallies[0]["end"]) == (1, 11)


def test_split_records_into_rallies_sorts_unsorted_input():
    obj = _bare_system()
    records = [{"frame": 3}, {"frame": 1}, {"frame": 2}]
    rallies = obj._split_records_into_rallies(records)
    assert len(rallies) == 1
    assert rallies[0]["frames"] == [1, 2, 3]


# ----------------------------------------------------------------------------
# _calibrate_camera multi-span drift orchestration (R12) — real CameraModel
# calibration against synthetic court-keypoint projections; fake video I/O.
# ----------------------------------------------------------------------------

def _lookat_extrinsics(cam_pos, target):
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    tvec = (-R @ cam_pos).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


def _project_court_keypoints(cam_pos, target):
    """构造一个真值 CameraModel，把 COURT_KEYPOINTS_M 投影成一组「检测到的」像素点
    （零噪声，14/14 全部有效）。与 tests/test_camera.py 的 _lookat_extrinsics 手法一致。"""
    K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
    rvec, tvec = _lookat_extrinsics(cam_pos, target)
    truth_camera = CameraModel(K, rvec, tvec)
    obj_points = np.column_stack([COURT_KEYPOINTS_M, np.zeros(len(COURT_KEYPOINTS_M))])
    return truth_camera.project(obj_points), truth_camera


_TARGET = np.array([5.485, 11.885, 1.0])
# 两个机位差异足够大（正对 vs 从底线另一侧看回来），确保漂移判定能稳定命中。
_BEFORE_POS = np.array([5.485, -6.0, 3.0])
_AFTER_POS = np.array([5.485, 29.0, 3.0])


class _FakeCap:
    """把 0-indexed 的 CAP_PROP_POS_FRAMES seek 位置喂给 provider(pos) 换关键点数组；
    provider 直接返回「检测结果」（跳过 resize/推理，_FakeDetector 原样透传）。"""

    def __init__(self, provider):
        self._provider = provider
        self._pos = 0

    def set(self, _prop, value):
        self._pos = value

    def read(self):
        return True, self._provider(self._pos)

    def release(self):
        pass


class _FakeCV2Module:
    CAP_PROP_POS_FRAMES = 1  # 任意占位值，_FakeCap.set 不区分 prop

    def __init__(self, provider):
        self._provider = provider

    def VideoCapture(self, _path):
        return _FakeCap(self._provider)


class _FakeDetector:
    """detect(frame) 原样返回 frame——frame 已经是 provider 给的关键点数组本身，
    不需要真的做图像推理。"""

    def __init__(self, _model_path):
        pass

    def detect(self, frame):
        return frame


def _patch_calibration_globals(monkeypatch, provider):
    # raising=False：这些名字在 system.py 里是 `global cv2` 等声明的模块级变量，
    # 只有真正跑过 load_runtime_dependencies() 之后才会作为模块属性存在；这里
    # 测试完全绕开那条重依赖加载路径，所以属性在 setattr 之前本就不存在。
    monkeypatch.setattr(system_module, "cv2", _FakeCV2Module(provider), raising=False)
    monkeypatch.setattr(system_module, "CourtKeypointDetector", _FakeDetector, raising=False)
    monkeypatch.setattr(system_module, "median_keypoints_over_frames", median_keypoints_over_frames, raising=False)
    monkeypatch.setattr(system_module, "keypoints_drifted", keypoints_drifted, raising=False)
    monkeypatch.setattr(system_module, "CameraModel", CameraModel, raising=False)
    monkeypatch.setattr(system_module, "COURT_KEYPOINTS_M", COURT_KEYPOINTS_M, raising=False)


def _calibration_ready_system(tmp_path, total_frames):
    keypoint_model_path = tmp_path / "fake_court_keypoints.pt"
    keypoint_model_path.touch()  # 只需要 os.path.exists 为真，内容不会被读取（detector 是假的）

    obj = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    obj.court_calibration = "keypoints"
    obj.keypoint_model_path = str(keypoint_model_path)
    obj.total_frames = total_frames
    obj.video_path = "unused-fake-video-path"
    obj.frame_width = W
    obj.frame_height = H
    return obj


def test_calibrate_camera_drift_triggers_recalibration_and_new_span(tmp_path, monkeypatch):
    before_points, before_truth = _project_court_keypoints(_BEFORE_POS, _TARGET)
    after_points, after_truth = _project_court_keypoints(_AFTER_POS, _TARGET)

    # 前置断言：确认两个机位的差异真的够大（>=4 点偏移 >10px），否则这个测试
    # 对被测的漂移判定逻辑没有区分力（会「假阳性通过」）。
    deltas = np.linalg.norm(before_points - after_points, axis=1)
    assert int(np.sum(deltas > 10.0)) >= 4

    def provider(pos_0indexed):
        # 初始采样窗口（0-indexed <900，对应 1-indexed 帧 1..291）恒为 before；
        # 漂移重检帧（0-indexed 900 = 1-indexed 901）与之后的重标定窗口恒为 after。
        return after_points if pos_0indexed >= 900 else before_points

    _patch_calibration_globals(monkeypatch, provider)
    obj = _calibration_ready_system(tmp_path, total_frames=1300)

    spans, recalibrated_at_frames = obj._calibrate_camera()

    assert recalibrated_at_frames == [901]
    assert len(spans) == 2
    assert spans[0]["start_frame"] == 1 and spans[0]["end_frame"] == 900
    assert spans[1]["start_frame"] == 901 and spans[1]["end_frame"] == 1300

    # 各段标定出的相机应分别还原出对应真值机位（不是随便一个能通过重投影误差检查
    # 的相机——验证漂移后确实切换到了「碰动后」的机位，不是继续用旧机位凑合）。
    air_point = np.array([[5.485, 11.885, 2.5]])
    assert np.linalg.norm(spans[0]["camera"].project(air_point)[0] - before_truth.project(air_point)[0]) < 8.0
    assert np.linalg.norm(spans[1]["camera"].project(air_point)[0] - after_truth.project(air_point)[0]) < 8.0

    # R12：_camera_for_frame 用真实产出的 spans 也能查到正确的段。
    assert TennisAnalysisSystem._camera_for_frame(spans, 500) is spans[0]["camera"]
    assert TennisAnalysisSystem._camera_for_frame(spans, 901) is spans[1]["camera"]


def test_calibrate_camera_no_drift_keeps_single_span_and_empty_recalibration_list(tmp_path, monkeypatch):
    before_points, _truth = _project_court_keypoints(_BEFORE_POS, _TARGET)

    def provider(_pos_0indexed):
        return before_points  # 全程同一机位——漂移守卫检到的应恒等于标定基线

    _patch_calibration_globals(monkeypatch, provider)
    obj = _calibration_ready_system(tmp_path, total_frames=1300)

    spans, recalibrated_at_frames = obj._calibrate_camera()

    assert recalibrated_at_frames == []
    assert len(spans) == 1
    assert spans[0]["start_frame"] == 1 and spans[0]["end_frame"] == 1300


def test_calibrate_camera_short_video_never_reaches_drift_check(tmp_path, monkeypatch):
    """demo.mp4 的真实情形（417 帧 < 901）：漂移检查帧号本身就超出视频总帧数，
    循环体一次都不会执行，spans 恒为单段、recalibrated_at_frames 恒为空——
    这正是 Step 3/4 端到端冒烟里实际观察到的行为，这里补一个不依赖真实视频/
    权重的确定性单测。"""
    before_points, _truth = _project_court_keypoints(_BEFORE_POS, _TARGET)

    def provider(_pos_0indexed):
        return before_points

    _patch_calibration_globals(monkeypatch, provider)
    obj = _calibration_ready_system(tmp_path, total_frames=417)

    spans, recalibrated_at_frames = obj._calibrate_camera()

    assert recalibrated_at_frames == []
    assert len(spans) == 1
    assert spans[0]["start_frame"] == 1 and spans[0]["end_frame"] == 417
