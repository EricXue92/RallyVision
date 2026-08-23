from tools.report_builder import ReportBuildError
from tools.worker import build_cli_args, pick_error_code


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
