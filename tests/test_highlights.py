from tennis_analysis.analysis.rally import Rally
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
