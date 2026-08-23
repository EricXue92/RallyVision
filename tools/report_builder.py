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


def _load_bounces(output_dir):
    path = os.path.join(output_dir, "detections.jsonl")
    bounces = []
    if not os.path.isfile(path):
        return bounces
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                # 真实文件尾部可能有截断行，跳过解析失败的行
                continue
            bounce = row.get("bounce")
            if not bounce:
                continue
            if "court" not in bounce:
                continue
            bounces.append({
                "frame": row.get("frame"),
                "time_sec": row.get("time_sec"),
                "court": bounce["court"],
                "line_call": bounce.get("line_call"),
            })
    return bounces


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

    return {
        "report_version": 1,
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
        "bounces": _load_bounces(output_dir),
    }


def has_highlights(output_dir):
    return os.path.isfile(os.path.join(output_dir, "highlights.mp4"))
