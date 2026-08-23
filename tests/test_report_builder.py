import json
import pytest
from tools.report_builder import ReportBuildError, build_report, has_highlights

_META = {"schema_version": "1.2",
         "video": {"fps": 60.0, "duration_sec": 10.0, "width": 1280, "height": 720},
         "camera": {"K": [[1.0]]}}
_SCORE = {"config": {}, "history": [], "score_timeline": [], "points": [], "final": "0-0 | 0-0"}
_STATS = {"players": {}, "rally_length_histogram": {}, "bounce_heatmap": {}}
_SHOTS = [{"hit_frame": 1, "line_call": "in", "shot_type": "serve"}]


def _write(dirpath, name, obj):
    (dirpath / name).write_text(json.dumps(obj), encoding="utf-8")


def _full_outputs(tmp_path):
    _write(tmp_path, "metadata.json", _META)
    _write(tmp_path, "match_score.json", _SCORE)
    _write(tmp_path, "match_stats.json", _STATS)
    _write(tmp_path, "shot_metrics.json", _SHOTS)
    lines = [
        {"frame": 5, "time_sec": 0.08, "bounce": None},
        {"frame": 25, "time_sec": 0.41, "bounce": {"court": [4.1, 5.2], "line_call": "out"}},
        {"frame": 40, "time_sec": 0.66, "bounce": {"court": [5.0, 20.0]}},  # 无 line_call
    ]
    (tmp_path / "detections.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    return tmp_path


def test_build_report_contract(tmp_path):
    report = build_report(str(_full_outputs(tmp_path)))
    assert report["report_version"] == 2
    assert report["schema_version"] == "1.2"
    assert report["precision_tier"] == "full"
    assert report["video"]["fps"] == 60.0
    assert report["match_score"]["final"] == "0-0 | 0-0"
    assert report["shots"][0]["shot_type"] == "serve"
    assert report["bounces"] == [
        {"frame": 25, "time_sec": 0.41, "court": [4.1, 5.2], "line_call": "out"},
        {"frame": 40, "time_sec": 0.66, "court": [5.0, 20.0], "line_call": None},
    ]
    assert report["player_movement"] is None


def test_homography_degrades_tier(tmp_path):
    d = _full_outputs(tmp_path)
    meta = dict(_META)
    meta["camera"] = None
    _write(d, "metadata.json", meta)
    assert build_report(str(d))["precision_tier"] == "line_call_only"


def test_missing_match_layer_is_court_not_detected(tmp_path):
    _write(tmp_path, "metadata.json", _META)
    with pytest.raises(ReportBuildError) as e:
        build_report(str(tmp_path))
    assert e.value.code == "court_not_detected"


def test_missing_metadata_is_pipeline_error(tmp_path):
    with pytest.raises(ReportBuildError) as e:
        build_report(str(tmp_path))
    assert e.value.code == "pipeline_error"


def test_has_highlights(tmp_path):
    assert has_highlights(str(tmp_path)) is False
    (tmp_path / "highlights.mp4").write_bytes(b"x")
    assert has_highlights(str(tmp_path)) is True


def test_player_movement_aggregation(tmp_path):
    d = _full_outputs(tmp_path)
    lines = [
        {"frame": 1, "time_sec": 0.0, "bounce": None,
         "players": {"lower": {"court": [5.0, 25.0], "speed": 0.0},
                     "upper": {"court": [5.0, 1.0], "speed": 2.0}}},
        {"frame": 2, "time_sec": 0.2, "bounce": None,
         "players": {"lower": {"court": [5.0, 24.0], "speed": 5.0},
                     "upper": {"court": [5.0, 40.0], "speed": 3.0}}},  # upper 越出网格
        {"frame": 3, "time_sec": 1.5, "bounce": None,                  # 断档 1.3s > 0.5s
         "players": {"lower": {"court": [5.0, 23.0], "speed": 6.0}}},
    ]
    (d / "detections.jsonl").write_text(
        "\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    report = build_report(str(d))
    assert report["bounces"] == []          # 本 fixture 无 bounce,重扫描不误伤既有聚合
    pm = report["player_movement"]
    assert pm["grid_m"] == 0.5 and pm["origin_xy"] == [-3.0, -5.0]
    assert pm["cols"] == 34 and pm["rows"] == 68
    lower = pm["players"]["lower"]
    # court [5.0, 25.0] → col=(5+3)/0.5=16, row=(25+5)/0.5=60
    assert lower["heat"][60][16] == 1
    assert lower["heat"][58][16] == 1       # [5.0, 24.0]
    assert lower["distance_m"] == 1.0       # 仅 0.0→0.2 帧计入:5.0 m/s × 0.2 s;断档帧跳过
    assert lower["max_speed_kmh"] == 21.6   # 6.0 m/s × 3.6
    upper = pm["players"]["upper"]
    assert sum(sum(r) for r in upper["heat"]) == 1   # 越界帧不进 heat
    assert upper["distance_m"] == 0.6                # 3.0 m/s × 0.2 s(越界帧仍计距离)
    assert upper["max_speed_kmh"] == 10.8


def test_player_movement_side_without_speed_has_null_max(tmp_path):
    d = _full_outputs(tmp_path)
    lines = [{"frame": 1, "time_sec": 0.0, "bounce": None,
              "players": {"lower": {"court": [5.0, 25.0], "speed": None}}}]
    (d / "detections.jsonl").write_text(json.dumps(lines[0]), encoding="utf-8")
    pm = build_report(str(d))["player_movement"]
    assert pm is not None                       # lower 有 court 帧,块存在
    assert pm["players"]["lower"]["max_speed_kmh"] is None
    assert pm["players"]["lower"]["distance_m"] == 0.0
    assert pm["players"]["upper"]["distance_m"] == 0.0   # 无帧一侧照常输出零值
