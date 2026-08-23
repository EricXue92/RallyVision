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
    assert report["report_version"] == 1
    assert report["schema_version"] == "1.2"
    assert report["precision_tier"] == "full"
    assert report["video"]["fps"] == 60.0
    assert report["match_score"]["final"] == "0-0 | 0-0"
    assert report["shots"][0]["shot_type"] == "serve"
    assert report["bounces"] == [
        {"frame": 25, "time_sec": 0.41, "court": [4.1, 5.2], "line_call": "out"},
        {"frame": 40, "time_sec": 0.66, "court": [5.0, 20.0], "line_call": None},
    ]


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
