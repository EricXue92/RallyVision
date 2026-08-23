"""方案②(视频自适应模板)单测:`_select_template_frame_from_video` 的选帧编排。

背景:is_court_view 原来拿写死的 templates/demo.png 做逐帧模板匹配,与 demo 图
观感差异大的视频(生产实测:美网蓝场转播)全部帧被拒 → 零检测。改为开场用
CourtKeypointDetector 在全片均匀采样、选有效关键点最多的帧当模板;全部检不出
或权重缺失回退 demo 旧路径(旧路径零球场帧已按 court_not_detected 收尾)。

测试手法与 test_system_orchestration.py 相同:bare `__new__` 实例 + monkeypatch
system.py 模块级 cv2 / CourtKeypointDetector 全局(raising=False,理由见彼处)。
"""
import numpy as np

import tennis_analysis.system as system_module
from tennis_analysis.system import TennisAnalysisSystem

W, H = 1280, 720


def _frame(tag):
    """用像素值编码帧身份,便于断言选中的是哪一帧。"""
    return np.full((H, W, 3), tag, dtype=np.uint8)


def _points(valid_count):
    """构造 [14,2] 关键点数组,前 valid_count 个点有效,其余 NaN。"""
    points = np.full((14, 2), np.nan, dtype=np.float32)
    points[:valid_count] = 100.0
    return points


class _FakeVideoCapture:
    """按 POS_FRAMES seek 返回预设帧;frames_by_index 是 {0-indexed 帧号: 帧}。"""

    def __init__(self, frames_by_index, total):
        self._frames = frames_by_index
        self._total = total
        self._pos = 0
        self.released = False

    def set(self, _prop, value):
        self._pos = int(value)

    def read(self):
        if self._pos >= self._total:
            return False, None
        frame = self._frames.get(self._pos)
        if frame is None:
            frame = _frame(0)
        self._pos += 1
        return True, frame

    def release(self):
        self.released = True

    def isOpened(self):
        return True


class _FakeCV2:
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, cap):
        self._cap = cap

    def VideoCapture(self, _path):
        return self._cap


class _FakeDetector:
    """按帧像素 tag 返回预设关键点;tag 不在表内返回 None(未检出球场)。"""

    points_by_tag = {}

    def __init__(self, _model_path):
        pass

    def detect(self, frame):
        return self.points_by_tag.get(int(frame[0, 0, 0]))


def _system(tmp_path, total_frames, cap):
    weights = tmp_path / "fake_court_keypoints.pt"
    weights.touch()  # 只需 os.path.exists 为真,fake detector 不读内容
    obj = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    obj.keypoint_model_path = str(weights)
    obj.total_frames = total_frames
    obj.video_path = "unused-fake-video-path"
    return obj


def _patch(monkeypatch, cap, points_by_tag):
    # np 与 cv2 同为 load_runtime_dependencies 懒注入的模块全局;这里接真 numpy
    monkeypatch.setattr(system_module, "np", np, raising=False)
    monkeypatch.setattr(system_module, "cv2", _FakeCV2(cap), raising=False)
    monkeypatch.setattr(_FakeDetector, "points_by_tag", points_by_tag)
    monkeypatch.setattr(system_module, "CourtKeypointDetector", _FakeDetector, raising=False)


def test_picks_frame_with_most_valid_keypoints(tmp_path, monkeypatch):
    # 帧 10 检出 8 点,帧 20 检出 12 点,其余检不出 → 应选帧 20
    cap = _FakeVideoCapture({10: _frame(1), 20: _frame(2)}, total=30)
    _patch(monkeypatch, cap, {1: _points(8), 2: _points(12)})
    obj = _system(tmp_path, total_frames=30, cap=cap)

    selected = obj._select_template_frame_from_video()

    assert selected is not None
    assert int(selected[0, 0, 0]) == 2
    assert cap.released


def test_full_14_keypoints_short_circuits(tmp_path, monkeypatch):
    # 帧 0 就是 14 点满配 → 直接选它,不再继续采样
    cap = _FakeVideoCapture({0: _frame(3), 10: _frame(4)}, total=30)
    _patch(monkeypatch, cap, {3: _points(14), 4: _points(14)})
    obj = _system(tmp_path, total_frames=30, cap=cap)

    selected = obj._select_template_frame_from_video()

    assert int(selected[0, 0, 0]) == 3


def test_no_court_in_any_frame_returns_none(tmp_path, monkeypatch):
    cap = _FakeVideoCapture({}, total=30)
    _patch(monkeypatch, cap, {})
    obj = _system(tmp_path, total_frames=30, cap=cap)

    assert obj._select_template_frame_from_video() is None
    assert cap.released


def test_missing_weights_returns_none_without_opening_video(tmp_path, monkeypatch):
    cap = _FakeVideoCapture({}, total=30)
    _patch(monkeypatch, cap, {})
    obj = _system(tmp_path, total_frames=30, cap=cap)
    obj.keypoint_model_path = str(tmp_path / "nonexistent.pt")

    assert obj._select_template_frame_from_video() is None
    assert not cap.released  # 权重缺失应直接返回,不该去开视频


def test_zero_total_frames_returns_none(tmp_path, monkeypatch):
    cap = _FakeVideoCapture({}, total=0)
    _patch(monkeypatch, cap, {})
    obj = _system(tmp_path, total_frames=0, cap=cap)

    assert obj._select_template_frame_from_video() is None
