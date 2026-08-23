"""比分修正重算工具 / Point-editing tool with full replay recalculation.

Task 8：改判 `match_score.json` 里的任意一分 → `MatchState` 从头重放 →
重写 `match_score.json`（history / score_timeline / final 全量重算）与
`match_stats.json` 的 `points_won` / `point_win_rate`。这是本产品对
SwingVision 的差异化（它不能改历史分），也是 iOS 端修正 UI 未来要调的
同一套逻辑，先以 CLI 收口。

Edits the winner of any recorded point, replays the whole match through the
scoring state machine, and rewrites `match_score.json` (fully recomputed
timeline/final) plus the `points_won` / `point_win_rate` fields of
`match_stats.json`.

用法 / usage：

    uv run tools/edit_point.py --output-dir outputs/demo --list
    uv run tools/edit_point.py --output-dir outputs/demo --point 12 --winner upper

注意 / notes：
- 改判可能让比赛提前结束；其后的分成为「垃圾分」——history/points/timeline
  长度不变（再次改判还能救回来），但垃圾分不计入 points_won / point_win_rate，
  timeline 快照冻结在终局比分（与 `MatchState.edit_point` 的忽略语义一致）。
- `point_win_rate` 的「该分包含哪些击球」用 shot_metrics 的 `hit_frame` 落在
  该分 `[start_frame, end_frame]` 帧区间内判定（rallies 未持久化，帧区间是
  同一信息的等价映射）；`count` 与其余统计块不属于本工具职责，原样保留。
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tennis_analysis.analysis.scoring import MatchState  # noqa: E402
from tennis_analysis.data.writer import write_json  # noqa: E402

SIDE_TO_INT = {"upper": 0, "lower": 1}
INT_TO_SIDE = {0: "upper", 1: "lower"}

# point_win_rate 样本阈值，与 stats_report.MIN_POINT_SAMPLE 同口径
# （不 import 是避免误导——本工具用帧区间而非 Rally 对象映射击球，口径独立成文）。
MIN_POINT_SAMPLE = 3

INITIAL_SCORE_LINE = "0-0 | 0-0"


def replay_timeline(config, history):
    """用公开 API 从头重放 history，返回逐分快照与「是否真正计分」标记。

    Returns: {"timeline": [score_line, ...], "counted": [bool, ...], "final": str}
    垃圾分（比赛已结束后的多余分）counted=False，其快照冻结在终局比分。
    """
    ms = MatchState(**config)
    timeline = []
    counted = []
    for record in history:
        try:
            ms.apply_point(record["winner"], record["reason"], record.get("rally_frames", []))
            counted.append(True)
        except ValueError:
            counted.append(False)
        timeline.append(ms.score_line())
    return {"timeline": timeline, "counted": counted, "final": ms.score_line()}


def apply_edit(match_score, point_idx, new_winner, reason="manual"):
    """改判第 point_idx 分给 new_winner（'upper'|'lower'），全量重算后返回
    新 match_score dict（不修改入参）。越界抛 IndexError，赢家非法抛 ValueError。
    """
    if new_winner not in SIDE_TO_INT:
        raise ValueError(
            "赢家只能是 upper 或 lower / winner must be 'upper' or 'lower', got %r" % (new_winner,)
        )
    history = match_score["history"]
    if not (0 <= point_idx < len(history)):
        raise IndexError(
            "分序号越界（共 %d 分）/ point index out of range (have %d points): %d"
            % (len(history), len(history), point_idx)
        )

    ms = MatchState.from_dict({"config": match_score["config"], "history": history})
    ms.edit_point(point_idx, SIDE_TO_INT[new_winner], reason)

    edited = dict(match_score)
    edited.update(ms.to_dict())
    replay = replay_timeline(edited["config"], edited["history"])
    edited["score_timeline"] = replay["timeline"]
    edited["final"] = replay["final"]

    points = [dict(point) for point in match_score.get("points", [])]
    if point_idx < len(points):
        points[point_idx]["winner"] = new_winner
        points[point_idx]["reason"] = reason
    edited["points"] = points
    return edited


def _point_categories(point, shots):
    """该分帧区间内每个击球者打出的类别集合（对齐 stats_report 的分内去重口径）。"""
    categories_by_hitter = {}
    for shot in shots:
        hit_frame = shot.get("hit_frame")
        if hit_frame is None or not (point["start_frame"] <= hit_frame <= point["end_frame"]):
            continue
        hitter = shot.get("hitter")
        if hitter not in SIDE_TO_INT:
            continue
        categories_by_hitter.setdefault(hitter, set()).add(shot.get("shot_type"))
    return categories_by_hitter


def update_stats(match_stats, match_score, shots):
    """按改判后的 match_score 重算 match_stats 的 points_won 与 point_win_rate
    （只动这两处，其余字段原样保留；垃圾分不计入）。返回新 dict，不修改入参。
    """
    replay = replay_timeline(match_score["config"], match_score["history"])
    points = match_score.get("points", [])

    points_won = {"upper": 0, "lower": 0}
    win_samples = {}  # (hitter, category) -> [bool, ...]
    for index, point in enumerate(points):
        if index >= len(replay["counted"]) or not replay["counted"][index]:
            continue
        winner = point["winner"]
        if winner in points_won:
            points_won[winner] += 1
        for hitter, categories in _point_categories(point, shots).items():
            for category in categories:
                win_samples.setdefault((hitter, category), []).append(hitter == winner)

    updated = json.loads(json.dumps(match_stats))  # 深拷贝，保住其余字段
    for side in ("upper", "lower"):
        player = updated["players"][side]
        player["points_won"] = points_won[side]
        for category, entry in player["shots"].items():
            samples = win_samples.get((side, category))
            if samples and len(samples) >= MIN_POINT_SAMPLE:
                entry["point_win_rate"] = float(sum(samples) / len(samples))
            else:
                entry["point_win_rate"] = None
    return updated


def format_point_lines(match_score):
    """--list 输出：每分一行（序号 / 帧区间 / 赢家 / reason / 该分开始前的比分）。"""
    timeline = match_score.get("score_timeline", [])
    lines = []
    for index, point in enumerate(match_score.get("points", [])):
        score_before = timeline[index - 1] if index > 0 else INITIAL_SCORE_LINE
        lines.append(
            "#%-3d 帧 %d-%d  赢家=%-5s  reason=%-12s  当时比分 [%s]"
            % (index, point["start_frame"], point["end_frame"],
               point["winner"], point["reason"], score_before)
        )
    return lines


def _atomic_write_json(path, payload):
    """tmp + os.replace 原子替换（与 system.py 同一套写法），防中途失败截断旧文件。"""
    tmp_path = "%s.tmp" % path
    write_json(tmp_path, payload)
    os.replace(tmp_path, path)


def _load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            raise SystemExit("找不到文件 / file not found: %s" % path)
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="改判任意一分并全量重算比分与统计 / edit any point and replay the whole match",
    )
    parser.add_argument("--output-dir", required=True,
                        help="pipeline 输出目录（含 match_score.json）/ pipeline output dir")
    parser.add_argument("--list", action="store_true",
                        help="打印每一分 / list every recorded point")
    parser.add_argument("--point", type=int, default=None,
                        help="要改判的分序号（--list 里的 # 序号）/ point index to edit")
    parser.add_argument("--winner", choices=("upper", "lower"), default=None,
                        help="改判后的赢家 / new winner")
    args = parser.parse_args(argv)

    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(REPO_ROOT, args.output_dir)
    match_score_path = os.path.join(output_dir, "match_score.json")
    match_stats_path = os.path.join(output_dir, "match_stats.json")
    shot_metrics_path = os.path.join(output_dir, "shot_metrics.json")

    match_score = _load_json(match_score_path)

    if args.list:
        lines = format_point_lines(match_score)
        if not lines:
            print("没有记录到任何分 / no points recorded")
        for line in lines:
            print(line)
        print("终局比分 / final: %s" % match_score.get("final", ""))
        if args.point is None:
            return 0

    if args.point is None or args.winner is None:
        parser.error("需要 --list，或 --point 与 --winner 成对出现 / "
                     "need --list, or both --point and --winner")

    try:
        edited = apply_edit(match_score, args.point, args.winner)
    except (IndexError, ValueError) as exc:
        raise SystemExit("改判失败 / edit failed: %s" % exc)

    _atomic_write_json(match_score_path, edited)
    print("已改判第 %d 分给 %s，比分已从头重放 / point #%d re-awarded to %s, match replayed"
          % (args.point, args.winner, args.point, args.winner))
    print("新终局比分 / new final: %s" % edited["final"])

    match_stats = _load_json(match_stats_path, required=False)
    if match_stats is None:
        print("未找到 match_stats.json，跳过统计重算 / match_stats.json missing, stats skipped")
        return 0
    shots = _load_json(shot_metrics_path, required=False) or []
    updated = update_stats(match_stats, edited, shots)
    _atomic_write_json(match_stats_path, updated)
    print("统计已重算（points_won / point_win_rate）/ stats recomputed: %s"
          % match_stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
