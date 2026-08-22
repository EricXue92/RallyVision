"""Task 11: 真实数据验证工具的纯逻辑单测 / unit tests for the validation tool's pure logic.

只测 manifest 解析 / 最近 fit_ok 段选取 / 误差与中位数计算 / 诊断统计（Step 3 用），
不测 subprocess 编排本身——那部分由 demo manifest 端到端跑一遍覆盖（见 task-11-report.md）。
"""
import json
import os

import pytest

from tools.validate_speed import (
    ManifestError,
    build_pipeline_args,
    compute_relative_error,
    load_manifest,
    main,
    median_relative_error,
    output_dir_for,
    rms_px_distribution,
    segment_visibility_stats,
    select_nearest_fit_ok_segment,
    trajectory_visibility_stats,
)


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

def _write(tmp_path, name, obj):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def test_load_manifest_parses_valid_entries(tmp_path):
    path = _write(tmp_path, "m.json", [
        {"video": "videos/demo.mp4", "hit_frame_approx": 360, "caption_kmh": 187},
        {"video": "videos/demo1.mp4", "hit_frame_approx": 10, "caption_kmh": 150.5, "label": "clip2"},
    ])
    entries = load_manifest(path)
    assert len(entries) == 2
    assert entries[0]["video"] == "videos/demo.mp4"
    assert entries[0]["hit_frame_approx"] == 360
    assert entries[0]["caption_kmh"] == 187.0
    assert entries[0]["label"] == "videos/demo.mp4"      # 无 label 时回退到 video 路径
    assert entries[1]["label"] == "clip2"


def test_load_manifest_rejects_non_list(tmp_path):
    path = _write(tmp_path, "m.json", {"video": "x"})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_missing_file_raises_manifest_error_not_raw_traceback(tmp_path):
    # review fix: 之前 open() 的 FileNotFoundError 会原样冒出去，不是工具自己的双语
    # ManifestError——manifest 路径打错是最常见的用户错误，必须走友好路径。
    missing_path = os.path.join(str(tmp_path), "does_not_exist.json")
    with pytest.raises(ManifestError):
        load_manifest(missing_path)


def test_load_manifest_malformed_json_raises_manifest_error(tmp_path):
    # review fix：json.JSONDecodeError 同样要归一到 ManifestError，不能裸传出去。
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json,,,")
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_rejects_empty_list(tmp_path):
    path = _write(tmp_path, "m.json", [])
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_rejects_missing_keys(tmp_path):
    path = _write(tmp_path, "m.json", [{"video": "videos/demo.mp4", "hit_frame_approx": 1}])
    with pytest.raises(ManifestError):
        load_manifest(path)


@pytest.mark.parametrize("bad_entry", [
    {"video": "", "hit_frame_approx": 1, "caption_kmh": 100},
    {"video": "v.mp4", "hit_frame_approx": -1, "caption_kmh": 100},
    {"video": "v.mp4", "hit_frame_approx": 1.5, "caption_kmh": 100},
    {"video": "v.mp4", "hit_frame_approx": 1, "caption_kmh": 0},
    {"video": "v.mp4", "hit_frame_approx": 1, "caption_kmh": -5},
])
def test_load_manifest_rejects_bad_values(tmp_path, bad_entry):
    path = _write(tmp_path, "m.json", [bad_entry])
    with pytest.raises(ManifestError):
        load_manifest(path)


# ---------------------------------------------------------------------------
# select_nearest_fit_ok_segment
# ---------------------------------------------------------------------------

def _entry(hit_frame, fit_ok, speed_kmh=100.0, rms_px=5.0):
    return {"hit_frame": hit_frame, "fit_ok": fit_ok, "speed_kmh": speed_kmh, "rms_px": rms_px}


def test_select_nearest_fit_ok_picks_closest_by_hit_frame():
    entries = [_entry(1, False), _entry(25, True), _entry(48, True), _entry(230, True)]
    picked = select_nearest_fit_ok_segment(entries, hit_frame_approx=50)
    assert picked["hit_frame"] == 48


def test_select_nearest_fit_ok_ignores_non_fit_ok():
    entries = [_entry(49, False), _entry(200, True)]
    picked = select_nearest_fit_ok_segment(entries, hit_frame_approx=50)
    assert picked["hit_frame"] == 200


def test_select_nearest_fit_ok_returns_none_when_all_fail():
    entries = [_entry(1, False), _entry(360, False)]
    assert select_nearest_fit_ok_segment(entries, hit_frame_approx=50) is None


def test_select_nearest_fit_ok_empty_list_returns_none():
    assert select_nearest_fit_ok_segment([], hit_frame_approx=50) is None


def test_select_nearest_fit_ok_tie_break_is_deterministic():
    # 200 与 50 同样距 approx=125 差 75，取较小 hit_frame 的一条（确定性优先于随机字典序）。
    entries = [_entry(200, True), _entry(50, True)]
    picked = select_nearest_fit_ok_segment(entries, hit_frame_approx=125)
    assert picked["hit_frame"] == 50


# ---------------------------------------------------------------------------
# compute_relative_error / median_relative_error
# ---------------------------------------------------------------------------

def test_compute_relative_error_basic():
    assert compute_relative_error(110.0, 100.0) == pytest.approx(0.10)
    assert compute_relative_error(90.0, 100.0) == pytest.approx(0.10)
    assert compute_relative_error(100.0, 100.0) == 0.0


def test_compute_relative_error_rejects_non_positive_caption():
    with pytest.raises(ValueError):
        compute_relative_error(100.0, 0.0)
    with pytest.raises(ValueError):
        compute_relative_error(100.0, -5.0)


def test_median_relative_error_empty_is_none():
    assert median_relative_error([]) is None


def test_median_relative_error_odd_and_even():
    assert median_relative_error([0.1, 0.2, 0.3]) == pytest.approx(0.2)
    assert median_relative_error([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# build_pipeline_args / output_dir_for
# ---------------------------------------------------------------------------

def test_build_pipeline_args_contains_required_flags():
    args = build_pipeline_args("videos/demo.mp4", "templates/demo.png", "outputs/x", ball_detector="tracknet", line_call="doubles")
    assert "--video-path" in args and args[args.index("--video-path") + 1] == "videos/demo.mp4"
    assert args[args.index("--ball-detector") + 1] == "tracknet"
    assert args[args.index("--output-dir") + 1] == "outputs/x"
    assert args[args.index("--shot-metrics") + 1] == "true"
    assert args[args.index("--line-call") + 1] == "doubles"
    assert args[args.index("--display") + 1] == "false"


def test_output_dir_for_keyed_by_video_stem():
    # 同一支视频重复验证跑要落到同一目录，才能复用缓存的 court_annotations.txt
    # （否则每次都会触发 main.py 的交互式球场确认阻塞）。
    d0 = output_dir_for("videos/demo.mp4", root="outputs/validate_speed")
    d1 = output_dir_for("videos/demo.mp4", root="outputs/validate_speed")
    assert d0 == d1
    assert d0.endswith(os.path.join("outputs", "validate_speed", "demo"))


def test_output_dir_for_different_videos_differ():
    d0 = output_dir_for("videos/demo.mp4", root="outputs/validate_speed")
    d1 = output_dir_for("videos/demo1.mp4", root="outputs/validate_speed")
    assert d0 != d1


def test_output_dir_for_default_root_matches_main_py_convention():
    # 默认根目录要落在 outputs/<stem>，与 main.py 自己单独跑该视频时的默认输出目录
    # 完全一致，这样同一支视频的缓存 court_annotations.txt 天然复用（见模块 docstring）。
    d = output_dir_for("videos/demo.mp4")
    assert d == os.path.join("outputs", "demo") or d.endswith(os.path.join("outputs", "demo"))


# ---------------------------------------------------------------------------
# 诊断统计（Step 3）：trajectory_visibility_stats / segment_visibility_stats / rms_px_distribution
# ---------------------------------------------------------------------------

def _pt(frame, image, interpolated=False):
    return {"frame": frame, "image": image, "interpolated": interpolated}


def test_trajectory_visibility_stats_counts_missing_and_interpolated():
    points = [
        _pt(0, None),                       # raw missing
        _pt(1, [1.0, 1.0]),                 # visible
        _pt(2, [2.0, 2.0], interpolated=True),  # interpolated gap
        _pt(3, [3.0, 3.0]),                 # visible
    ]
    stats = trajectory_visibility_stats(points)
    assert stats["total_frames"] == 4
    assert stats["visible_frames"] == 2
    assert stats["visible_pct"] == pytest.approx(50.0)
    assert stats["raw_missing_frames"] == 1
    assert stats["interpolated_frames"] == 1
    assert stats["gap_runs"] == [1, 1]
    assert stats["max_gap_run"] == 1


def test_trajectory_visibility_stats_run_length_encoding():
    points = [_pt(i, None) for i in range(3)] + [_pt(3, [0.0, 0.0])] + [_pt(4, None), _pt(5, None)]
    stats = trajectory_visibility_stats(points)
    assert stats["gap_runs"] == [3, 2]
    assert stats["max_gap_run"] == 3


def test_segment_visibility_stats_within_window_only():
    points = [_pt(i, [float(i), 0.0]) for i in range(10)]
    points[5]["image"] = None
    stats = segment_visibility_stats(points, hit_frame=4, bounce_frame=7)
    assert stats["n_frames"] == 4
    assert stats["gap_frames"] == 1
    assert stats["raw_missing_frames"] == 1
    assert stats["gap_pct"] == pytest.approx(25.0)


def test_segment_visibility_stats_frame_absent_from_trajectory_counts_as_missing():
    points = [_pt(0, [0.0, 0.0]), _pt(2, [2.0, 0.0])]   # frame 1 缺失（连点都没有,不只是 image=None)
    stats = segment_visibility_stats(points, hit_frame=0, bounce_frame=2)
    assert stats["n_frames"] == 3
    assert stats["raw_missing_frames"] == 1


def test_rms_px_distribution_basic():
    entries = [
        {"fit_ok": False, "rms_px": 42.33},
        {"fit_ok": False, "rms_px": 19.31},
        {"fit_ok": True, "rms_px": 5.0},
    ]
    stats = rms_px_distribution(entries, threshold=12.0)
    assert stats["count"] == 3
    assert stats["min"] == 5.0
    assert stats["max"] == 42.33
    assert stats["median"] == pytest.approx(19.31)
    assert stats["over_threshold"] == 2
    assert stats["fit_ok_count"] == 1
    assert stats["fit_fail_count"] == 2


def test_rms_px_distribution_handles_none_values():
    entries = [{"fit_ok": False, "rms_px": None}]
    stats = rms_px_distribution(entries)
    assert stats["count"] == 0
    assert stats["min"] is None
    assert stats["fit_fail_count"] == 1


# ---------------------------------------------------------------------------
# main() 顶层错误处理（review fix：missing/malformed manifest 不能裸 traceback）
# ---------------------------------------------------------------------------

def test_main_missing_manifest_file_returns_error_code_not_traceback(tmp_path, capsys):
    missing_path = os.path.join(str(tmp_path), "does_not_exist.json")
    rc = main(["--manifest", missing_path])
    assert rc == 1
    out = capsys.readouterr().out
    assert "manifest" in out.lower()


def test_main_malformed_manifest_returns_error_code_not_traceback(tmp_path, capsys):
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json,,,")
    rc = main(["--manifest", path])
    assert rc == 1
    out = capsys.readouterr().out
    assert "manifest" in out.lower()
