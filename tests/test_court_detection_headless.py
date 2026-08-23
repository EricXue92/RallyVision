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
