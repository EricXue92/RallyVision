import subprocess

from tennis_analysis.analysis.rally import Rally
from tennis_analysis.export import highlights
from tennis_analysis.export.highlights import export_highlights, select_highlight_rallies


def _rally(start_frame, end_frame, n_shots):
    return Rally(
        start_frame=start_frame,
        end_frame=end_frame,
        shots=[
            {"hit_frame": start_frame + i * 5, "shot_type": "forehand", "hitter": "lower"}
            for i in range(n_shots)
        ],
        bounces=[],
        end_reason="video_end",
    )


def test_long_rally_selected_by_shot_count():
    long_rally = _rally(0, 100, n_shots=9)
    points = [
        {
            "winner": "lower",
            "reason": "unknown",
            "start_frame": 0,
            "end_frame": 100,
            "serve_number": 1,
            "rally_indices": [0],
        }
    ]

    selected = select_highlight_rallies([long_rally], points, min_shots=8)

    assert selected == [long_rally]


def test_short_winner_point_rally_selected():
    short_rally = _rally(0, 40, n_shots=3)
    points = [
        {
            "winner": "lower",
            "reason": "winner",
            "start_frame": 0,
            "end_frame": 40,
            "serve_number": 1,
            "rally_indices": [0],
        }
    ]

    selected = select_highlight_rallies([short_rally], points, min_shots=8)

    assert selected == [short_rally]


def test_ordinary_short_rally_not_selected():
    ordinary_rally = _rally(0, 40, n_shots=3)
    points = [
        {
            "winner": "lower",
            "reason": "out",
            "start_frame": 0,
            "end_frame": 40,
            "serve_number": 1,
            "rally_indices": [0],
        }
    ]

    selected = select_highlight_rallies([ordinary_rally], points, min_shots=8)

    assert selected == []


def test_select_preserves_order_over_mixed_rallies():
    long_rally = _rally(0, 100, n_shots=9)
    ordinary_rally = _rally(150, 190, n_shots=3)
    winner_rally = _rally(200, 240, n_shots=2)
    points = [
        {
            "winner": "lower", "reason": "unknown", "start_frame": 0, "end_frame": 100,
            "serve_number": 1, "rally_indices": [0],
        },
        {
            "winner": "lower", "reason": "out", "start_frame": 150, "end_frame": 190,
            "serve_number": 1, "rally_indices": [1],
        },
        {
            "winner": "lower", "reason": "winner", "start_frame": 200, "end_frame": 240,
            "serve_number": 1, "rally_indices": [2],
        },
    ]

    selected = select_highlight_rallies(
        [long_rally, ordinary_rally, winner_rally], points, min_shots=8
    )

    assert selected == [long_rally, winner_rally]


def test_export_highlights_returns_false_when_ffmpeg_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: None)

    rally = _rally(0, 100, n_shots=9)
    ok = export_highlights(
        video_path="videos/demo.mp4",
        rallies=[rally],
        score_lines=["0-0"],
        out_path=str(tmp_path / "highlight.mp4"),
        fps=25.0,
    )

    captured = capsys.readouterr()
    assert ok is False
    assert "ffmpeg" in captured.out
    # 双语警告：中文 + English 都应出现
    assert any("一" <= ch <= "鿿" for ch in captured.out)


def test_export_highlights_degrades_when_drawtext_unavailable(monkeypatch, capsys, tmp_path):
    # ffmpeg 本身"存在"，但探测到没编译 drawtext 滤镜
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(highlights, "_drawtext_available", lambda ffmpeg_bin: False)

    recorded_cmds = []

    def fake_run(cmd, capture_output=True, text=True):
        recorded_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(highlights.subprocess, "run", fake_run)

    rally = _rally(0, 100, n_shots=9)
    out_path = tmp_path / "highlight.mp4"
    ok = export_highlights(
        video_path="videos/demo.mp4",
        rallies=[rally],
        score_lines=["0-0"],
        out_path=str(out_path),
        fps=25.0,
    )

    captured = capsys.readouterr()
    assert ok is True
    # 无 ffmpeg 真调用（fake_run 打桩），仍应看到降级警告（双语）
    assert "drawtext" in captured.out
    assert any("一" <= ch <= "鿿" for ch in captured.out)

    # 切片命令里不应再出现 -vf/drawtext（降级为纯裁剪，无叠加）
    segment_cmds = [cmd for cmd in recorded_cmds if "-f" not in cmd]
    assert segment_cmds, "expected at least one segment-cut command to be recorded"
    for cmd in segment_cmds:
        assert "-vf" not in cmd
        assert not any("drawtext" in str(arg) for arg in cmd)


def test_export_highlights_uses_drawtext_filter_when_available(monkeypatch, tmp_path):
    # ffmpeg 存在,且探测到 drawtext 可用 → 切片命令应带 -vf drawtext=...
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(highlights, "_drawtext_available", lambda ffmpeg_bin: True)

    recorded_cmds = []

    def fake_run(cmd, capture_output=True, text=True):
        recorded_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(highlights.subprocess, "run", fake_run)

    rally = _rally(0, 100, n_shots=9)
    score_text = "O'Brien 6-4"  # 含单引号,验证真的走了转义,不是巧合过关
    out_path = tmp_path / "highlight.mp4"
    ok = export_highlights(
        video_path="videos/demo.mp4",
        rallies=[rally],
        score_lines=[score_text],
        out_path=str(out_path),
        fps=25.0,
    )

    assert ok is True

    segment_cmds = [cmd for cmd in recorded_cmds if "-f" not in cmd]
    assert segment_cmds, "expected at least one segment-cut command to be recorded"
    cmd = segment_cmds[0]
    assert "-vf" in cmd

    vf_value = cmd[cmd.index("-vf") + 1]
    expected_escaped = highlights._escape_drawtext(score_text)
    expected_filter = (
        f"drawtext=text='{expected_escaped}':x=24:y=24:fontsize=36:"
        "fontcolor=white:box=1:boxcolor=black@0.5"
    )
    assert vf_value == expected_filter
    # 断言真的转义过(单引号真被换成了收尾/转义/重开的写法),不是原样漏转义
    assert "'\\''" in vf_value
    assert "O'Brien" not in vf_value


def test_escape_drawtext_escapes_single_quote_with_breakout_idiom():
    # 单引号需要「收尾引号 + 转义插入字面引号 + 重开引号」的 '\'' 写法
    assert highlights._escape_drawtext("O'Brien") == "O'\\''Brien"


def test_escape_drawtext_leaves_colon_and_backslash_literal():
    # 一旦被外层单引号包住,ffmpeg 自己的 filtergraph 解析器逐字复制引号内内容,
    # 冒号、反斜杠都不再有特殊含义,不应被转义
    assert highlights._escape_drawtext("6:4 3\\2") == "6:4 3\\2"


def test_escape_drawtext_treats_none_as_empty_string():
    assert highlights._escape_drawtext(None) == ""


def test_escape_concat_path_escapes_single_quote():
    assert (
        highlights._escape_concat_path("/tmp/seg's/seg_0.mp4")
        == "/tmp/seg'\\''s/seg_0.mp4"
    )


def test_drawtext_available_false_when_subprocess_raises_oserror(monkeypatch):
    def raising_run(cmd, capture_output=True, text=True):
        raise OSError("no such file or directory")

    monkeypatch.setattr(highlights.subprocess, "run", raising_run)

    assert highlights._drawtext_available("/usr/bin/ffmpeg") is False


def test_drawtext_available_false_when_unknown_filter_reported(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="Unknown filter 'drawtext'.\n"
        )

    monkeypatch.setattr(highlights.subprocess, "run", fake_run)

    assert highlights._drawtext_available("/usr/bin/ffmpeg") is False


def test_drawtext_available_true_when_filter_help_printed(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True):
        return subprocess.CompletedProcess(
            cmd, returncode=0,
            stdout="Filter drawtext\n  Draw text on top of video frames...\n",
            stderr="",
        )

    monkeypatch.setattr(highlights.subprocess, "run", fake_run)

    assert highlights._drawtext_available("/usr/bin/ffmpeg") is True
