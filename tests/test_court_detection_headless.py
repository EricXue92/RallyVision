"""Phase 4a worker 冒烟测试踩到的坑:`--display false` 只挡了主处理窗口的
`cv2.waitKey(1)`(system.py 里的 `self.show_display` 分支),没挡 court 自动检测
的确认弹窗——`_confirm_auto_court_detection` 不看 `show_display`,一律
`cv2.namedWindow`+`cv2.imshow`+阻塞 `cv2.waitKey(0)` 等 Enter/Y/M/R/Esc。

worker 每个 job 的 `--output-dir` 按 job_id 分,`court_annotations.txt` 缓存
(system.py `_setup_court_annotation`)永远命中不到上一次,所以 worker 起的每一单
只要跑到球场检测这步,不管检测成不成功,都会在这里死等键盘输入——本地冒烟两次
在这卡了快 25 分钟才用 `sample`(macOS 采样工具)抓到堆栈定位到 `cv::waitKey`。

跟 `tests/test_system_orchestration.py` 一样的手法:`TennisAnalysisSystem.__new__`
跳过 `__init__`(不用真视频/权重),monkeypatch `system.py` 的模块级懒加载全局
(`cv2`/`np`/`CourtLineAutoDetector`),只是这次用一个「记录用了哪些交互式 cv2 调用」
的 cv2 代理,而不是完全假的 cv2——图像运算(polylines/putText/imwrite 等)照样走真
cv2,只拦截 waitKey/imshow/namedWindow 三个会弹窗/阻塞的调用。
"""
import numpy as np
import pytest

import tennis_analysis.system as system_module
from tennis_analysis.system import TennisAnalysisSystem


class _RecordingCV2Proxy:
    """真 cv2 的透传代理,只记录/拦截会弹窗或阻塞的三个调用。

    `waitKey` 若真被调用,返回 13(Enter/接受)而不是让它挂起——这样即使修复前的
    bug 复现了也不会真的把测试进程焊死,只是 `wait_key_calls` 非空,足以证明
    「本该跳过但没跳过」。
    """

    def __init__(self, real_cv2):
        self._real = real_cv2
        self.wait_key_calls = []
        self.imshow_calls = []
        self.named_window_calls = []

    def waitKey(self, delay):
        self.wait_key_calls.append(delay)
        return 13

    def imshow(self, *args, **kwargs):
        self.imshow_calls.append((args, kwargs))

    def namedWindow(self, *args, **kwargs):
        self.named_window_calls.append((args, kwargs))

    def destroyWindow(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FakeAutoDetector:
    """跳过真推理,直接原样返回构造时传入的检测结果。"""

    def __init__(self, result):
        self._result = result
        self.last_diagnostics = {}

    def detect(self, _template_color):
        return self._result


def _headless_system(tmp_path):
    system = TennisAnalysisSystem.__new__(TennisAnalysisSystem)
    system.show_display = False
    system.court_detection = "auto-fallback"
    system.save_dir = str(tmp_path)
    system.court_detection_result = None
    return system


def _detected_result():
    return {
        "corners": [(10, 10), (630, 10), (630, 350), (10, 350)],
        "roi_corners": [(0, 0), (640, 360)],
        "mid_height": 180,
        "line_count": 8,
        "confidence": 0.9,
        "diagnostics": {},
    }


def test_headless_auto_detection_skips_confirmation_prompt(monkeypatch, tmp_path):
    detected = _detected_result()
    proxy = _RecordingCV2Proxy(system_module.cv2 if hasattr(system_module, "cv2") else __import__("cv2"))
    monkeypatch.setattr(system_module, "cv2", proxy, raising=False)
    monkeypatch.setattr(system_module, "np", np, raising=False)
    monkeypatch.setattr(
        system_module, "CourtLineAutoDetector",
        lambda: _FakeAutoDetector(detected), raising=False,
    )

    system = _headless_system(tmp_path)
    template_color = np.zeros((360, 640, 3), dtype=np.uint8)

    corners, roi_corners, mid_height = system._detect_or_annotate_court(template_color)

    assert corners == detected["corners"]
    assert roi_corners == detected["roi_corners"]
    assert mid_height == detected["mid_height"]
    assert system.court_detection_result["accepted"] is True
    # 核心回归断言:headless 模式绝不能碰这三个会弹窗/阻塞的 cv2 调用
    assert proxy.wait_key_calls == []
    assert proxy.imshow_calls == []
    assert proxy.named_window_calls == []


def test_headless_with_no_detection_raises_instead_of_blocking_on_manual_click(monkeypatch, tmp_path):
    proxy = _RecordingCV2Proxy(system_module.cv2 if hasattr(system_module, "cv2") else __import__("cv2"))
    monkeypatch.setattr(system_module, "cv2", proxy, raising=False)
    monkeypatch.setattr(system_module, "np", np, raising=False)
    monkeypatch.setattr(
        system_module, "CourtLineAutoDetector",
        lambda: _FakeAutoDetector(None), raising=False,
    )

    system = _headless_system(tmp_path)
    template_color = np.zeros((360, 640, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="headless"):
        system._detect_or_annotate_court(template_color)

    assert proxy.wait_key_calls == []
    assert proxy.named_window_calls == []


def _candidate_result():
    """detector.detect() 判定不够置信,靠 diagnostics 兜底出来的低置信度候选结果——
    交互模式下这本来就是要弹窗人工确认的那一档,跟 detected 分支不是同一回事。"""
    return {
        "corners": [(5, 5), (600, 5), (600, 300), (5, 300)],
        "roi_corners": [(0, 0), (600, 300)],
        "mid_height": 150,
        "line_count": 3,
        "confidence": 0.31,
        "diagnostics": {},
    }


def test_headless_low_confidence_candidate_raises_instead_of_auto_accepting(monkeypatch, tmp_path):
    """Review finding(coordinator-mandated 修复):headless 只能自动采信过了
    detector min_confidence 门槛的 `detected` 分支;`candidate` 是交互模式下本来就
    要弹窗人工确认的低置信度兜底结果,report.json 冻结契约里没有 confidence/status
    字段区分好坏检测,headless 下绝不能像 `detected` 分支一样静默采信——必须显式
    失败(用户拿 error_code 去重录),而不是产出一份可能全错但看起来正常的报告。"""
    candidate = _candidate_result()
    proxy = _RecordingCV2Proxy(system_module.cv2 if hasattr(system_module, "cv2") else __import__("cv2"))
    monkeypatch.setattr(system_module, "cv2", proxy, raising=False)
    monkeypatch.setattr(system_module, "np", np, raising=False)
    monkeypatch.setattr(
        system_module, "CourtLineAutoDetector",
        lambda: _FakeAutoDetector(None), raising=False,
    )

    system = _headless_system(tmp_path)
    # detect() 返回 None,让流程走到 candidate = _candidate_from_diagnostics(...)；
    # 直接在实例上打桩这个方法,跳过它内部真实的 compute_expanded_roi 角点数学
    # (跟这条回归无关,不需要真的算)。
    monkeypatch.setattr(system, "_candidate_from_diagnostics", lambda _tc, _diag: candidate)
    template_color = np.zeros((360, 640, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match=r"[Ll]ow confidence|置信度偏低"):
        system._detect_or_annotate_court(template_color)

    # 跟坑 2 的修复一样:headless 下绝不能弹窗/阻塞等键盘
    assert proxy.wait_key_calls == []
    assert proxy.imshow_calls == []
    assert proxy.named_window_calls == []
    assert system.court_detection_result["accepted"] is False
    assert system.court_detection_result["confidence"] == candidate["confidence"]


def test_headless_high_confidence_detected_still_auto_accepts(monkeypatch, tmp_path):
    """跟上面低置信度的 candidate 分支对比:detected 分支已经过 detector 的
    min_confidence 门槛,headless 下应该继续保持之前的行为——直接采信,不因为这次
    review 收紧了 candidate 分支就连带把 detected 分支也改严。"""
    detected = _detected_result()
    proxy = _RecordingCV2Proxy(system_module.cv2 if hasattr(system_module, "cv2") else __import__("cv2"))
    monkeypatch.setattr(system_module, "cv2", proxy, raising=False)
    monkeypatch.setattr(system_module, "np", np, raising=False)
    monkeypatch.setattr(
        system_module, "CourtLineAutoDetector",
        lambda: _FakeAutoDetector(detected), raising=False,
    )

    system = _headless_system(tmp_path)
    template_color = np.zeros((360, 640, 3), dtype=np.uint8)

    corners, roi_corners, mid_height = system._detect_or_annotate_court(template_color)

    assert corners == detected["corners"]
    assert system.court_detection_result["status"] == "auto"
    assert system.court_detection_result["accepted"] is True
    assert proxy.wait_key_calls == []


def test_interactive_low_confidence_candidate_still_prompts_for_human_confirmation(monkeypatch, tmp_path):
    """Coordinator ruling:交互模式的人工确认弹窗必须原样保留,不能被这次收紧误伤。"""
    candidate = _candidate_result()
    proxy = _RecordingCV2Proxy(system_module.cv2 if hasattr(system_module, "cv2") else __import__("cv2"))
    monkeypatch.setattr(system_module, "cv2", proxy, raising=False)
    monkeypatch.setattr(system_module, "np", np, raising=False)
    monkeypatch.setattr(
        system_module, "CourtLineAutoDetector",
        lambda: _FakeAutoDetector(None), raising=False,
    )

    system = _headless_system(tmp_path)
    system.show_display = True  # 交互模式
    monkeypatch.setattr(system, "_candidate_from_diagnostics", lambda _tc, _diag: candidate)
    template_color = np.zeros((360, 640, 3), dtype=np.uint8)

    corners, roi_corners, mid_height = system._detect_or_annotate_court(template_color)

    # proxy.waitKey 固定返回 13(Enter/接受)——证明弹窗确认流程真的跑了一遍
    # (namedWindow/imshow/waitKey 都被调用),而不是被短路跳过。
    assert corners == candidate["corners"]
    assert proxy.wait_key_calls == [0]
    assert proxy.named_window_calls
    assert system.court_detection_result["status"] == "auto_low_confidence_accepted"
    assert system.court_detection_result["accepted"] is True
