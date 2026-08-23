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
