"""Task 10：纯后处理链 segments -> metrics -> spin -> line_call -> 写 shot_metrics.json。

不跑视频；数据合成方式复用 tests/test_segments.py（points/bounce_events/player_positions
字典结构）与 tests/test_trajectory3d.py 的 `_make_camera`（真实相机 + 物理仿真弹道，
确保 fit_ok=True 分支也被覆盖，不只是退化路径）。

这里练的 tennis_analysis.analysis.shot_pipeline.compute_shot_metrics_entries 必须与
system.py `_cleanup` 实际调用的是同一份代码（Step 6 binding），不许 system.py 另抄一份链。
"""
import json

import numpy as np

from tennis_analysis.analysis.physics import sample_at, simulate_trajectory
from tennis_analysis.analysis.shot_pipeline import (
    compute_shot_metrics_entries,
    write_shot_metrics,
)
from tennis_analysis.data.writer import SCHEMA_VERSION
from tests.test_trajectory3d import _make_camera

SCHEMA_KEYS = {
    "hit_frame", "bounce_frame", "hitter", "speed_kmh", "spin_coeff",
    "fit_ok", "rms_px", "spin_label", "spin_confidence", "line_call",
}


def _make_shot(v0=(0.5, 28.0, 1.5), spin=0.0, fps=60.0, noise_px=1.0, seed=3):
    """单拍合成弹道：hit(frame 0) -> bounce(最后一帧)，无需换向即可配对成一段。"""
    cam = _make_camera()
    p0 = np.array([5.0, 2.5, 1.1])
    times, pos = simulate_trajectory(p0, np.array(v0), spin, duration=1.2)
    landing = np.argmax(pos[1:, 2] < 0.033) + 1
    t_land = times[landing]
    # 落地帧之后再多采 8 帧（真实 cleaned_ball_trajectory 弹跳后球仍继续飞），
    # 否则 classify_spin 的 post_bounce_vy 窗口内没有数据可算，spin_label 恒 None。
    t_post = min(times[-1], t_land + 8.0 / fps)
    frame_times = np.arange(0, t_post, 1.0 / fps)
    pts3d = sample_at(times, pos, frame_times)
    px = cam.project(pts3d)
    rng = np.random.default_rng(seed)
    px = px + rng.normal(0, noise_px, px.shape)
    bounce_xy = pos[landing][:2]
    bounce_frame_index = int(round(t_land * fps))

    points = [
        {
            "frame": index,
            "time_sec": round(float(t), 6),
            "image": [float(px[index, 0]), float(px[index, 1])],
            "court": [float(pts3d[index, 0]), float(pts3d[index, 1])],
        }
        for index, t in enumerate(frame_times)
    ]
    bounce_events = [{"frame": bounce_frame_index, "court": [float(bounce_xy[0]), float(bounce_xy[1])]}]
    player_positions = {"upper": [5.0, 2.5], "lower": [5.485, 21.0]}
    return cam, points, bounce_events, player_positions


def test_pipeline_produces_full_schema_with_camera():
    cam, points, bounce_events, player_positions = _make_shot()
    entries = compute_shot_metrics_entries(
        points, bounce_events, player_positions, fps=60.0, camera=cam, line_call_mode="doubles"
    )

    assert len(entries) == 1
    entry = entries[0]
    assert set(entry.keys()) == SCHEMA_KEYS
    assert entry["fit_ok"] is True
    assert entry["speed_kmh"] is not None
    assert entry["spin_coeff"] is not None
    assert entry["spin_label"] in {"topspin", "flat", "slice"}
    assert 0.0 <= entry["spin_confidence"] <= 1.0
    assert entry["line_call"] in {"in", "out", "close"}


def test_degraded_camera_none_keeps_only_line_call():
    _cam, points, bounce_events, player_positions = _make_shot()
    entries = compute_shot_metrics_entries(
        points, bounce_events, player_positions, fps=60.0, camera=None, line_call_mode="doubles"
    )

    assert len(entries) == 1
    entry = entries[0]
    assert set(entry.keys()) == SCHEMA_KEYS
    assert entry["fit_ok"] is False
    assert entry["speed_kmh"] is None
    assert entry["spin_coeff"] is None
    assert entry["spin_label"] is None
    assert entry["spin_confidence"] is None
    assert entry["rms_px"] is None
    assert entry["line_call"] in {"in", "out", "close"}   # 降级模式 line_call 仍可算


def test_line_call_off_skips_verdict():
    cam, points, bounce_events, player_positions = _make_shot()
    entries = compute_shot_metrics_entries(
        points, bounce_events, player_positions, fps=60.0, camera=cam, line_call_mode="off"
    )
    assert entries[0]["line_call"] is None


def test_inf_rms_sanitized_to_null_through_full_pipeline(tmp_path):
    # 只保留前 3 帧 image，其余置 None -> fit_segment 内 valid<8 提前返回 rms_px=inf。
    cam, points, bounce_events, player_positions = _make_shot()
    for point in points[3:]:
        point["image"] = None

    entries = compute_shot_metrics_entries(
        points, bounce_events, player_positions, fps=60.0, camera=cam, line_call_mode="doubles"
    )
    assert len(entries) == 1
    assert entries[0]["fit_ok"] is False
    assert entries[0]["rms_px"] is None  # inf 已转 null，不是残留 inf

    path = tmp_path / "shot_metrics.json"
    write_shot_metrics(str(path), entries)
    raw_text = path.read_text()
    assert "Infinity" not in raw_text  # 若未转换，json.dump 默认会写出非法 JSON 的 Infinity
    payload = json.loads(raw_text)
    assert payload[0]["rms_px"] is None


def test_write_shot_metrics_file_exists_and_schema_complete(tmp_path):
    cam, points, bounce_events, player_positions = _make_shot()
    entries = compute_shot_metrics_entries(
        points, bounce_events, player_positions, fps=60.0, camera=cam, line_call_mode="doubles"
    )
    path = tmp_path / "shot_metrics.json"
    write_shot_metrics(str(path), entries)

    assert path.exists()
    payload = json.loads(path.read_text())
    assert len(payload) == 1
    assert set(payload[0].keys()) == SCHEMA_KEYS


def test_schema_version_is_1_1():
    assert SCHEMA_VERSION == "1.1"
