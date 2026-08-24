import httpx
import pytest

from tools.report_builder import ReportBuildError
from tools.worker import build_cli_args, pick_error_code, pipeline_timeout, with_retry


def test_build_cli_args_defaults():
    args = build_cli_args("/tmp/v.mp4", "/tmp/out", None)
    assert args[:2] == ["uv", "run"]
    joined = " ".join(args)
    assert "--video-path /tmp/v.mp4" in joined
    assert "--output-dir /tmp/out" in joined
    assert "--match-scoring true" in joined
    assert "--highlights true" in joined
    assert "--line-call singles" in joined       # 第一版恒单打
    assert "--display false" in joined
    assert "--first-server lower" in joined      # 默认


def test_build_cli_args_params_override():
    args = " ".join(build_cli_args("/v.mp4", "/o", {
        "first_server": "upper", "best_of": 5, "no_ad": True,
        "upper_hand": "left", "lower_hand": "right",
    }))
    assert "--first-server upper" in args
    assert "--best-of 5" in args
    assert "--no-ad true" in args
    assert "--upper-hand left" in args


def test_build_cli_args_ignores_unknown_params():
    args = " ".join(build_cli_args("/v.mp4", "/o", {"evil": "--rm -rf"}))
    assert "evil" not in args and "--rm" not in args   # 白名单外的键静默忽略


def test_pick_error_code():
    assert pick_error_code(ReportBuildError("court_not_detected")) == "court_not_detected"
    assert pick_error_code(RuntimeError("boom")) == "pipeline_error"


def _http_status_error(status_code):
    request = httpx.Request("POST", "http://worker.invalid/x")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_with_retry_succeeds_after_transient_transport_errors():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.TransportError("connection reset")
        return "ok"

    assert with_retry(flaky, sleep_fn=sleeps.append) == "ok"
    assert calls["n"] == 3
    assert sleeps == [5, 15]           # 退避序列 5s → 15s


def test_with_retry_retries_5xx_http_status_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_status_error(502)
        return "ok"

    assert with_retry(flaky, sleep_fn=lambda s: None) == "ok"
    assert calls["n"] == 2


def test_with_retry_does_not_retry_4xx():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise _http_status_error(409)          # 如 claim token 失效,业务错误不重试

    with pytest.raises(httpx.HTTPStatusError):
        with_retry(bad, sleep_fn=lambda s: None)
    assert calls["n"] == 1                       # 只调用一次,没重试


def test_with_retry_raises_after_exhausting_attempts():
    sleeps = []

    def always_fails():
        raise httpx.TransportError("down")

    with pytest.raises(httpx.TransportError):
        with_retry(always_fails, sleep_fn=sleeps.append)
    assert sleeps == [5, 15]


def test_pipeline_timeout_formula(monkeypatch):
    monkeypatch.delenv("RV_PIPELINE_TIMEOUT", raising=False)
    assert pipeline_timeout(60) == 60 * 40 + 1800
    assert pipeline_timeout(0) == 1800


def test_pipeline_timeout_cap(monkeypatch):
    monkeypatch.delenv("RV_PIPELINE_TIMEOUT", raising=False)
    assert pipeline_timeout(10_000) == 72000     # 封顶 72000s(20h)


def test_pipeline_timeout_env_override(monkeypatch):
    monkeypatch.setenv("RV_PIPELINE_TIMEOUT", "999")
    assert pipeline_timeout(60) == 999
    assert pipeline_timeout(100_000) == 999      # 覆盖后不再受封顶公式约束


# ---- 標註影片:轉碼 + 流式上傳 / annotated video: transcode + streaming upload ----

from tools.worker import put_file_with_retry, transcode_annotated


def test_transcode_annotated_missing_source_returns_none(tmp_path):
    assert transcode_annotated(str(tmp_path)) is None


def test_transcode_annotated_success_returns_720p_path(tmp_path, monkeypatch):
    src = tmp_path / "detect_input.mp4"
    src.write_bytes(b"fake")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        (tmp_path / "annotated_720p.mp4").write_bytes(b"out")   # 實現校驗產物落盤
        class R: returncode = 0
        return R()

    monkeypatch.setattr("tools.worker.subprocess.run", fake_run)
    out = transcode_annotated(str(tmp_path))
    assert out == str(tmp_path / "annotated_720p.mp4")
    joined = " ".join(captured["cmd"])
    assert "scale=-2:720" in joined and "libx264" in joined
    assert "+faststart" in joined                  # 邊下邊播必需 moov 前置


def test_transcode_annotated_failure_degrades_to_source(tmp_path, monkeypatch):
    src = tmp_path / "detect_input.mp4"
    src.write_bytes(b"fake")

    def boom(cmd, **kwargs):
        raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr("tools.worker.subprocess.run", boom)
    assert transcode_annotated(str(tmp_path)) == str(src)


def test_put_file_with_retry_reopens_file_each_attempt(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"content-bytes")
    reads, calls = [], {"n": 0}

    def fake_put(url, content=None, headers=None, timeout=None):
        calls["n"] += 1
        reads.append(content.read())               # 消耗掉句柄,模擬真實流式上傳
        if calls["n"] == 1:
            raise httpx.TransportError("reset mid-upload")
        request = httpx.Request("PUT", url)
        return httpx.Response(200, request=request)

    put_file_with_retry("http://cos.invalid/x", str(f), "video/mp4",
                        timeout=5.0, put_fn=fake_put, sleep_fn=lambda s: None)
    assert calls["n"] == 2
    assert reads == [b"content-bytes", b"content-bytes"]   # 重試讀到的是完整文件,不是耗盡的句柄


def test_put_file_with_retry_sends_content_type(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    seen = {}

    def fake_put(url, content=None, headers=None, timeout=None):
        seen["headers"] = headers
        content.read()
        request = httpx.Request("PUT", url)
        return httpx.Response(200, request=request)

    put_file_with_retry("http://cos.invalid/x", str(f), "video/mp4",
                        timeout=5.0, put_fn=fake_put, sleep_fn=lambda s: None)
    assert seen["headers"] == {"Content-Type": "video/mp4"}
