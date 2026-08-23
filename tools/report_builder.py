"""Aggregate pipeline output-dir files into a single frozen-contract report.json dict.

聚合分析流水线的输出目录为单一的、契约冻结的 report.json dict。
"""

import json
import os


class ReportBuildError(Exception):
    """code in {"court_not_detected", "pipeline_error"}."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _read_json(output_dir, name):
    path = os.path.join(output_dir, name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 网格常数(与 spec §1 / iOS 契约一致,勿改)
_MOVE_GRID_M = 0.5
_MOVE_ORIGIN = (-3.0, -5.0)   # x 两侧各 3m、y 两端各 5m 边距,覆盖底线外站位(实测 y 可到 28m)
_MOVE_COLS = 34               # x ∈ [-3, 14)
_MOVE_ROWS = 68               # y ∈ [-5, 29)
_MOVE_MAX_DT = 0.5            # 与上一有效帧间隔超过 0.5s 视为断档(回合间无检测),不计跑动


def _empty_side():
    return {"heat": [[0] * _MOVE_COLS for _ in range(_MOVE_ROWS)],
            "distance_m": 0.0, "max_speed_ms": None, "last_time": None, "seen": False}


def _scan_detections(output_dir):
    """单遍扫 detections.jsonl,同时聚合 bounces 与 player_movement。"""
    path = os.path.join(output_dir, "detections.jsonl")
    bounces = []
    sides = {"upper": _empty_side(), "lower": _empty_side()}
    if not os.path.isfile(path):
        return bounces, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                # 真实文件尾部可能有截断行,跳过解析失败的行
                continue
            _collect_bounce(row, bounces)
            _collect_movement(row, sides)
    return bounces, _movement_out(sides)


def _collect_bounce(row, bounces):
    bounce = row.get("bounce")
    if not bounce or "court" not in bounce:
        return
    bounces.append({"frame": row.get("frame"), "time_sec": row.get("time_sec"),
                    "court": bounce["court"], "line_call": bounce.get("line_call")})


def _collect_movement(row, sides):
    time_sec = row.get("time_sec")
    players = row.get("players") or {}
    for name, agg in sides.items():
        p = players.get(name)
        if not p:
            continue
        court = p.get("court")
        if court and len(court) >= 2:
            agg["seen"] = True
            col = int((court[0] - _MOVE_ORIGIN[0]) / _MOVE_GRID_M)
            row_i = int((court[1] - _MOVE_ORIGIN[1]) / _MOVE_GRID_M)
            if 0 <= col < _MOVE_COLS and 0 <= row_i < _MOVE_ROWS:
                agg["heat"][row_i][col] += 1
        speed = p.get("speed")
        if speed is None or time_sec is None:
            continue
        if agg["max_speed_ms"] is None or speed > agg["max_speed_ms"]:
            agg["max_speed_ms"] = speed
        if agg["last_time"] is not None:
            dt = time_sec - agg["last_time"]
            if 0 < dt <= _MOVE_MAX_DT:
                agg["distance_m"] += speed * dt
        agg["last_time"] = time_sec


def _movement_out(sides):
    if not any(agg["seen"] for agg in sides.values()):
        return None
    players = {}
    for name, agg in sides.items():
        max_kmh = None if agg["max_speed_ms"] is None else round(agg["max_speed_ms"] * 3.6, 1)
        players[name] = {"heat": agg["heat"],
                         "distance_m": round(agg["distance_m"], 1),
                         "max_speed_kmh": max_kmh}
    return {"grid_m": _MOVE_GRID_M, "origin_xy": list(_MOVE_ORIGIN),
            "cols": _MOVE_COLS, "rows": _MOVE_ROWS, "players": players}


def build_report(output_dir):
    metadata = _read_json(output_dir, "metadata.json")
    if metadata is None:
        raise ReportBuildError("pipeline_error")

    match_score = _read_json(output_dir, "match_score.json")
    match_stats = _read_json(output_dir, "match_stats.json")
    if match_score is None or match_stats is None:
        raise ReportBuildError("court_not_detected")

    shots = _read_json(output_dir, "shot_metrics.json")
    video = metadata["video"]
    bounces, player_movement = _scan_detections(output_dir)

    return {
        "report_version": 2,   # 4c: + player_movement(纯增量,旧 iOS 忽略未知键)
        "schema_version": metadata["schema_version"],
        "video": {
            "fps": video["fps"],
            "duration_sec": video["duration_sec"],
            "width": video["width"],
            "height": video["height"],
        },
        "precision_tier": "full" if metadata.get("camera") is not None else "line_call_only",
        "match_score": match_score,
        "match_stats": match_stats,
        "shots": shots,
        "bounces": bounces,
        "player_movement": player_movement,
    }


def has_highlights(output_dir):
    return os.path.isfile(os.path.join(output_dir, "highlights.mp4"))
