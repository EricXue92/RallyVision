"""Task 8: 修正重算工具的纯逻辑单测 / unit tests for the point-editing tool.

只测核心函数 apply_edit / update_stats / format_point_lines（不测 argparse 壳，
CLI 由 Step 3 真实冒烟覆盖）。口径对齐 test_scoring.py 的 edit_point 测试：
改一分后 final / score_timeline 与「从头手工推演」完全一致。
"""
import pytest

from tennis_analysis.analysis.scoring import MatchState
from tools.edit_point import apply_edit, format_point_lines, replay_timeline, update_stats


def _build_match_score(point_specs, server=1):
    """按 match_layer.run_match_layer 的输出结构手造 match_score dict。

    point_specs: [(winner_side, reason, [start_frame, end_frame]), ...]
    """
    side_to_int = {"upper": 0, "lower": 1}
    ms = MatchState(server=server)
    timeline = []
    for index, (side, reason, frames) in enumerate(point_specs):
        ms.apply_point(side_to_int[side], reason, rally_frames=[index])
        timeline.append(ms.score_line())
    data = ms.to_dict()
    data["score_timeline"] = timeline
    data["points"] = [
        {
            "winner": side,
            "reason": reason,
            "start_frame": frames[0],
            "end_frame": frames[1],
            "serve_number": 1,
            "rally_indices": [index],
        }
        for index, (side, reason, frames) in enumerate(point_specs)
    ]
    data["final"] = ms.score_line()
    return data


def _specs_win(side, n, start=0):
    """连续 n 分同一赢家，帧区间递增（每分 100 帧）。"""
    return [
        (side, "winner", [start + i * 100, start + i * 100 + 90])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# apply_edit
# ---------------------------------------------------------------------------

def test_apply_edit_matches_hand_replayed_reference():
    # 与 test_scoring.test_edit_point_replays_everything_downstream 同口径：
    # upper 连拿 4 局（16 分）+ lower 拿 1 局（4 分），改第 0 分给 lower。
    specs = _specs_win("upper", 16) + _specs_win("lower", 4, start=1600)
    match_score = _build_match_score(specs)

    edited = apply_edit(match_score, 0, "lower")

    # 手工推演参照：第 0 分给 lower(=1)，其余按原 history 重放
    ref = MatchState(server=1)
    ref.apply_point(1, "manual")
    for record in match_score["history"][1:]:
        ref.apply_point(record["winner"], record["reason"], record["rally_frames"])
    assert edited["final"] == ref.score_line()

    # history 第 0 条改判 + reason 变 manual，rally_frames 保留
    assert edited["history"][0]["winner"] == 1
    assert edited["history"][0]["reason"] == "manual"
    assert edited["history"][0]["rally_frames"] == match_score["history"][0]["rally_frames"]
    assert len(edited["history"]) == 20

    # points 第 0 条同步改判（字符串口径），其余分原样
    assert edited["points"][0]["winner"] == "lower"
    assert edited["points"][0]["reason"] == "manual"
    assert edited["points"][1] == match_score["points"][1]


def test_apply_edit_recomputes_full_timeline():
    specs = _specs_win("upper", 16) + _specs_win("lower", 4, start=1600)
    match_score = _build_match_score(specs)

    edited = apply_edit(match_score, 0, "lower")

    # score_timeline 全量重算：与逐分重放快照逐项一致
    ref = MatchState(server=1)
    expected = []
    for record in edited["history"]:
        ref.apply_point(record["winner"], record["reason"], record["rally_frames"])
        expected.append(ref.score_line())
    assert edited["score_timeline"] == expected
    assert edited["score_timeline"][0] == "0-0 | 0-15"
    assert len(edited["score_timeline"]) == len(edited["history"])


def _garbage_specs():
    """构造「改判会让比赛提前结束」的局面：upper 连拿 47 分到 6-0 5-0 40-0
    赛点，lower 连救 5 分破发（40-40 → AD → 局）到 5-1，upper 再拿 4 分收下
    6-1 结束（共 56 分）。把第 47 分（lower 救的第一个赛点）改判给 upper →
    第 48 分就 6-0 6-0 打完，末尾 8 分成为「垃圾分」。"""
    return (
        _specs_win("upper", 47)
        + _specs_win("lower", 5, start=4700)
        + _specs_win("upper", 4, start=5200)
    )


def test_apply_edit_early_finish_keeps_garbage_points_frozen():
    # 垃圾分：history/points/timeline 长度不变（再次改判还能救回来），
    # 垃圾分的 timeline 快照冻结在终局比分（与 MatchState.edit_point 忽略语义一致）。
    match_score = _build_match_score(_garbage_specs())

    edited = apply_edit(match_score, 47, "upper")

    assert edited["final"] == "6-0 6-0 | 0-0 | 0-0"
    assert len(edited["score_timeline"]) == 56
    assert edited["score_timeline"][-8:] == [edited["final"]] * 8


def test_apply_edit_rejects_bad_input():
    match_score = _build_match_score(_specs_win("upper", 4))
    with pytest.raises(IndexError):
        apply_edit(match_score, 99, "lower")
    with pytest.raises(ValueError):
        apply_edit(match_score, 0, "left-player")


# ---------------------------------------------------------------------------
# replay_timeline
# ---------------------------------------------------------------------------

def test_replay_timeline_reports_counted_flags():
    match_score = _build_match_score(_garbage_specs())
    edited = apply_edit(match_score, 47, "upper")

    result = replay_timeline(edited["config"], edited["history"])
    assert result["final"] == edited["final"]
    assert result["timeline"] == edited["score_timeline"]
    assert result["counted"][:48] == [True] * 48
    assert result["counted"][48:] == [False] * 8


# ---------------------------------------------------------------------------
# update_stats
# ---------------------------------------------------------------------------

def _stats_skeleton():
    """match_stats.json 里本工具会动的最小结构（其余字段原样保留）。"""
    def player():
        return {
            "shots": {
                category: {"count": 0, "point_win_rate": None}
                for category in ("serve", "overhead", "volley", "forehand", "backhand", "unknown")
            },
            "serve": {"first_in_pct": None, "double_faults": 0,
                      "avg_speed_kmh": None, "max_speed_kmh": None},
            "points_won": 0,
        }
    return {
        "players": {"upper": player(), "lower": player()},
        "rally_length_histogram": {"1-3": 0, "4-6": 0, "7-9": 0, "10+": 0},
        "bounce_heatmap": {"grid_m": 1.0, "upper_half": [], "lower_half": []},
    }


def test_update_stats_recomputes_points_won_and_win_rate():
    # 4 分：lower 赢 3、upper 赢 1；每分里 lower 都打了 serve（帧区间内各 1 拍）
    specs = [
        ("lower", "winner", [0, 90]),
        ("lower", "winner", [100, 190]),
        ("upper", "winner", [200, 290]),
        ("lower", "winner", [300, 390]),
    ]
    match_score = _build_match_score(specs)
    shots = [
        {"hit_frame": 10, "hitter": "lower", "shot_type": "serve"},
        {"hit_frame": 110, "hitter": "lower", "shot_type": "serve"},
        {"hit_frame": 210, "hitter": "lower", "shot_type": "serve"},
        {"hit_frame": 310, "hitter": "lower", "shot_type": "serve"},
        {"hit_frame": 220, "hitter": "upper", "shot_type": "forehand"},
    ]
    stats = _stats_skeleton()
    stats["players"]["lower"]["points_won"] = 3
    stats["players"]["upper"]["points_won"] = 1
    stats["players"]["lower"]["shots"]["serve"]["point_win_rate"] = 0.75
    stats["players"]["lower"]["shots"]["serve"]["count"] = 4

    # 把第 2 分（upper 赢）改判给 lower → lower 4 胜 0 负
    edited = apply_edit(match_score, 2, "lower")
    updated = update_stats(stats, edited, shots)

    assert updated["players"]["lower"]["points_won"] == 4
    assert updated["players"]["upper"]["points_won"] == 0
    assert updated["players"]["lower"]["shots"]["serve"]["point_win_rate"] == 1.0
    # forehand 只出现在 1 分里（样本 < 3）→ None
    assert updated["players"]["upper"]["shots"]["forehand"]["point_win_rate"] is None
    # count 不属于本工具职责，原样保留
    assert updated["players"]["lower"]["shots"]["serve"]["count"] == 4
    # 其余块原样保留
    assert updated["rally_length_histogram"] == stats["rally_length_histogram"]


def test_update_stats_excludes_garbage_points():
    # 改判导致提前结束时，垃圾分不进 points_won
    match_score = _build_match_score(_garbage_specs())
    edited = apply_edit(match_score, 47, "upper")

    updated = update_stats(_stats_skeleton(), edited, shots=[])
    assert updated["players"]["upper"]["points_won"] == 48
    assert updated["players"]["lower"]["points_won"] == 0


# ---------------------------------------------------------------------------
# format_point_lines (--list)
# ---------------------------------------------------------------------------

def test_format_point_lines_one_line_per_point():
    specs = [
        ("lower", "out", [0, 90]),
        ("upper", "winner", [100, 190]),
    ]
    match_score = _build_match_score(specs)
    lines = format_point_lines(match_score)
    assert len(lines) == 2
    # 序号 / 帧区间 / 赢家 / reason / 当时比分（该分开始前的记分牌）
    assert "0" in lines[0] and "lower" in lines[0] and "out" in lines[0]
    assert "0-90" in lines[0]
    assert "0-0 | 0-0" in lines[0]          # 第 0 分开始前
    assert "0-0 | 0-15" in lines[1]         # 第 1 分开始前 = 第 0 分结束后
