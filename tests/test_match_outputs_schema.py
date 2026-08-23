"""Task 7：比赛层纯后处理链（shot_type -> rally -> point_outcome -> MatchState ->
stats -> 写 match_score.json / match_stats.json）单测，不跑视频。

手造 detections.jsonl 等价的逐帧记录（dict，键=帧号）+ shot_metrics 条目（阶段 2
产物，未附加 shot_type），验证 tennis_analysis.analysis.match_layer.run_match_layer
产出的两份 JSON 字段齐全、SCHEMA_VERSION == "1.2"、score_timeline 长度与 points
数一致。

场景（fps=10，BALL_LOST_FRAMES 按 fps 缩放阈值 = round(45*10/60) = 8 帧）：
- 回合 A（frame 1 发球 lower，frame 2 出界弹跳）：单拍发球 + out 弹跳 = 一发失误
  （fault），不单独出分，等下一回合配对。
- 回合 B（frame 20 发球 lower，frame 22 回击 upper，frame 24/26 同侧连续两次 in
  弹跳无回击 -> double_bounce 收尾）：与回合 A 的 fault 配对成一分（serve_number=2）。
- 回合 C（frame 40 发球 upper，无弹跳，video_end 收尾）：独立一分。
共 2 分，score_timeline 应有 2 条。
"""
import json

from tennis_analysis.analysis.match_layer import run_match_layer
from tennis_analysis.data.writer import SCHEMA_VERSION, write_json

FPS = 10.0


def _player(court=None, image=None, hands=None):
    return {
        "image": image,
        "court": court,
        "hands": hands or {"left": None, "right": None},
    }


def _empty_players():
    return {"upper": _player(), "lower": _player()}


def _make_detections():
    detections = {}

    # frame 1：第一拍，lower 在底线后发球（court y=23.0 > 23.77-1.0）
    detections[1] = {
        "players": {
            **_empty_players(),
            "lower": _player(
                court=[5.5, 23.0], image=[650, 900],
                hands={"left": [600.0, 850.0], "right": [700.0, 850.0]},
            ),
        },
        "tennis_ball": {"image": [650, 950]},
    }
    # frame 2：出界弹跳，upper 半场（y=4.0 < 11.885）
    detections[2] = {
        "players": _empty_players(),
        "tennis_ball": {"image": [650, 300]},
        "bounce": {"court": [5.5, 4.0], "line_call": "out"},
    }
    # frame 20：二发（与 frame 1 间隔 19 帧 >= 阈值 8，heuristic 判首拍）
    detections[20] = {
        "players": {
            **_empty_players(),
            "lower": _player(
                court=[5.5, 23.0], image=[650, 900],
                hands={"left": [600.0, 850.0], "right": [700.0, 850.0]},
            ),
        },
        "tennis_ball": {"image": [650, 950]},
    }
    # frame 22：upper 回击（间隔 2 帧 < 阈值，非首拍）
    detections[22] = {
        "players": {
            **_empty_players(),
            "upper": _player(
                court=[5.5, 20.0], image=[650, 450],
                hands={"left": [600.0, 400.0], "right": [700.0, 400.0]},
            ),
        },
        "tennis_ball": {"image": [760, 450]},
    }
    # frame 24 / 26：同侧（upper）连续两次界内弹跳、中间无回击 -> double_bounce
    detections[24] = {
        "players": _empty_players(),
        "tennis_ball": {"image": [650, 300]},
        "bounce": {"court": [5.5, 3.0], "line_call": "in"},
    }
    detections[26] = {
        "players": _empty_players(),
        "tennis_ball": {"image": [650, 300]},
        "bounce": {"court": [5.5, 2.5], "line_call": "in"},
    }
    # frame 40：独立一分的发球（upper，底线后 y=0.3 < 1.0），无后续事件 -> video_end
    detections[40] = {
        "players": {
            **_empty_players(),
            "upper": _player(
                court=[5.5, 0.3], image=[650, 50],
                hands={"left": [600.0, 0.0], "right": [700.0, 0.0]},
            ),
        },
        "tennis_ball": {"image": [650, 10]},
    }
    return detections


def _make_shot_metrics():
    # 阶段 2 shot_metrics.json 条目：故意不带 "shot_type"，match_layer 应只增字段。
    return [
        {"hit_frame": 1, "bounce_frame": 2, "hitter": "lower", "speed_kmh": 150.0,
         "spin_coeff": 0.1, "fit_ok": True, "rms_px": 1.2, "spin_label": "flat",
         "spin_confidence": 0.8, "line_call": "out"},
        {"hit_frame": 20, "bounce_frame": None, "hitter": "lower", "speed_kmh": 145.0,
         "spin_coeff": 0.1, "fit_ok": True, "rms_px": 1.1, "spin_label": "flat",
         "spin_confidence": 0.7, "line_call": None},
        {"hit_frame": 22, "bounce_frame": 24, "hitter": "upper", "speed_kmh": 90.0,
         "spin_coeff": 0.2, "fit_ok": True, "rms_px": 1.3, "spin_label": "topspin",
         "spin_confidence": 0.6, "line_call": "in"},
        {"hit_frame": 40, "bounce_frame": None, "hitter": "upper", "speed_kmh": 155.0,
         "spin_coeff": 0.05, "fit_ok": True, "rms_px": 1.0, "spin_label": "flat",
         "spin_confidence": 0.9, "line_call": None},
    ]


def test_run_match_layer_attaches_shot_type_without_dropping_fields():
    shot_metrics = _make_shot_metrics()
    detections = _make_detections()

    result = run_match_layer(
        shot_metrics, detections, FPS,
        first_server="lower", upper_hand="right", lower_hand="right",
        sets_to_win=2, no_ad=False,
    )

    updated = result["shot_metrics"]
    assert len(updated) == len(shot_metrics)
    for original, entry in zip(shot_metrics, updated):
        assert "shot_type" in entry
        for key, value in original.items():
            assert entry[key] == value


def test_run_match_layer_points_and_score_timeline_aligned():
    shot_metrics = _make_shot_metrics()
    detections = _make_detections()

    result = run_match_layer(
        shot_metrics, detections, FPS,
        first_server="lower", upper_hand="right", lower_hand="right",
        sets_to_win=2, no_ad=False,
    )

    points = result["points"]
    match_score = result["match_score"]

    assert len(points) == 2
    assert len(match_score["score_timeline"]) == len(points)
    assert match_score["points"] == points
    assert "final" in match_score
    assert "config" in match_score and "history" in match_score
    assert len(result["rally_score_lines"]) == len(result["rallies"])


def test_run_match_layer_stats_schema():
    shot_metrics = _make_shot_metrics()
    detections = _make_detections()

    result = run_match_layer(
        shot_metrics, detections, FPS,
        first_server="lower", upper_hand="right", lower_hand="right",
        sets_to_win=2, no_ad=False,
    )

    stats = result["match_stats"]
    assert set(stats.keys()) == {"players", "rally_length_histogram", "bounce_heatmap"}
    assert set(stats["players"].keys()) == {"upper", "lower"}
    total_points_won = sum(player["points_won"] for player in stats["players"].values())
    assert total_points_won == len(result["points"])


def test_written_json_files_have_schema_1_2(tmp_path):
    shot_metrics = _make_shot_metrics()
    detections = _make_detections()

    result = run_match_layer(
        shot_metrics, detections, FPS,
        first_server="lower", upper_hand="right", lower_hand="right",
        sets_to_win=2, no_ad=False,
    )

    match_score_path = tmp_path / "match_score.json"
    match_stats_path = tmp_path / "match_stats.json"
    write_json(str(match_score_path), result["match_score"])
    write_json(str(match_stats_path), result["match_stats"])

    assert SCHEMA_VERSION == "1.2"

    saved_score = json.loads(match_score_path.read_text())
    saved_stats = json.loads(match_stats_path.read_text())

    assert len(saved_score["score_timeline"]) == len(saved_score["points"])
    assert "players" in saved_stats
