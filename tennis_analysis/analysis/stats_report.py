"""比赛统计聚合（task-5-brief）：把 Rally 列表 + infer_points 输出 + shot_metrics
列表聚合成 match_stats.json 结构（六类击球胜率 / 发球统计 / 回合长度直方图 /
弹跳热图）。纯 dict/list 聚合 + 少量 numpy 网格，无 cv2/torch 依赖。

口径（binding，见 task-5-brief）：
- `players.<p>.shots.<category>.count`：该球员该类击球的总拍数（不限定是否落在
  某一分内，来自传入的 `shots` 全量列表）。
- `point_win_rate`：该球员该类击球所在分（经 `point["rally_indices"]` 找到对应
  回合的 shots 判定是否命中该类别）的胜率；样本（命中该类别的分数）< 3 时为
  None（防噪声）。
- `serve.first_in_pct`：serve_number==1 的分记一发在界内；serve_number==2 的分
  （无论是否 double_fault）记一发 fault，仍计入尝试分母。发球者取该分第一个
  回合的第一拍击球者（该拍恒为一发）。
- `serve.avg/max_speed_kmh`：只统计 `fit_ok` 为真的发球类型拍（来自 `shots` 全量
  列表，不限定是否落在某一分内）。
- `rally_length_histogram`：桶 = 该分所有回合 shots 数之和。
- `bounce_heatmap`：各回合全部弹跳（不限定是否落在某一分内）按 court 坐标落进
  1m 网格计数，按落点 y 是否 < 半场长（11.885m）分 upper_half/lower_half；
  出界坐标夹进边界格（不丢弃，网格总和恒等于弹跳总数）。
"""
import math
from collections import defaultdict

from .shot_type import SHOT_TYPES

PLAYERS = ("upper", "lower")

COURT_WIDTH_M = 10.97
FULL_COURT_LENGTH_M = 23.77
HALF_COURT_LENGTH_M = FULL_COURT_LENGTH_M / 2.0  # 11.885

GRID_M = 1.0
GRID_COLS = 11   # ceil(10.97 / 1m)
GRID_ROWS = 12   # ceil(11.885 / 1m)

HISTOGRAM_BUCKETS = ("1-3", "4-6", "7-9", "10+")
MIN_POINT_SAMPLE = 3  # point_win_rate 低于此样本数判 None


def _flip(side):
    return "lower" if side == "upper" else "upper"


def _histogram_bucket(total_shots):
    if total_shots <= 3:
        return "1-3"
    if total_shots <= 6:
        return "4-6"
    if total_shots <= 9:
        return "7-9"
    return "10+"


def _clamp_index(value, size):
    """把坐标值夹进 [0, size-1] 的网格下标（越界坐标落进边界格，不丢弃）。"""
    index = int(math.floor(value))
    return max(0, min(size - 1, index))


def _empty_grid():
    return [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]


def _new_player_stats():
    return {
        "shots": {
            category: {"count": 0, "point_win_rate": None} for category in SHOT_TYPES
        },
        "serve": {
            "first_in_pct": None,
            "double_faults": 0,
            "avg_speed_kmh": None,
            "max_speed_kmh": None,
        },
        "points_won": 0,
    }


def build_stats(rallies, points, shots) -> dict:
    players = {player: _new_player_stats() for player in PLAYERS}

    # 1) 击球分类计数：全量 shots，不限定是否落在某一分内。
    for shot in shots:
        hitter = shot.get("hitter")
        category = shot.get("shot_type")
        if hitter not in players or category not in SHOT_TYPES:
            continue
        players[hitter]["shots"][category]["count"] += 1

    # 2) 逐分聚合：回合长度直方图 + 发球一发口径 + point_win_rate 样本。
    histogram = {bucket: 0 for bucket in HISTOGRAM_BUCKETS}
    serve_attempts = defaultdict(int)
    serve_first_in = defaultdict(int)
    double_faults = defaultdict(int)
    win_samples = defaultdict(lambda: defaultdict(list))  # win_samples[hitter][category] -> [bool,...]

    for point in points:
        rally_indices = point["rally_indices"]
        point_rallies = [rallies[index] for index in rally_indices]

        total_shots = sum(len(rally.shots) for rally in point_rallies)
        histogram[_histogram_bucket(total_shots)] += 1

        server = point_rallies[0].shots[0]["hitter"]
        serve_attempts[server] += 1
        if point["serve_number"] == 1:
            serve_first_in[server] += 1
        if point["serve_number"] == 2 and point["reason"] == "double_fault":
            double_faults[server] += 1

        categories_by_hitter = defaultdict(set)
        for rally in point_rallies:
            for shot in rally.shots:
                categories_by_hitter[shot["hitter"]].add(shot.get("shot_type"))

        winner = point["winner"]
        for hitter, categories in categories_by_hitter.items():
            if hitter not in players:
                continue
            for category in categories:
                if category not in SHOT_TYPES:
                    continue
                win_samples[hitter][category].append(hitter == winner)

        if winner in players:
            players[winner]["points_won"] += 1

    for hitter, by_category in win_samples.items():
        for category, wins in by_category.items():
            if len(wins) < MIN_POINT_SAMPLE:
                continue
            players[hitter]["shots"][category]["point_win_rate"] = float(
                sum(1 for won in wins if won) / len(wins)
            )

    for player in PLAYERS:
        attempts = serve_attempts.get(player, 0)
        if attempts > 0:
            players[player]["serve"]["first_in_pct"] = float(
                serve_first_in.get(player, 0) / attempts
            )
        players[player]["serve"]["double_faults"] = int(double_faults.get(player, 0))

    # 3) 发球速度：只统计 fit_ok 为真的发球类型拍，来自全量 shots。
    serve_speeds = defaultdict(list)
    for shot in shots:
        if shot.get("shot_type") != "serve" or not shot.get("fit_ok"):
            continue
        speed = shot.get("speed_kmh")
        hitter = shot.get("hitter")
        if speed is None or hitter not in players:
            continue
        serve_speeds[hitter].append(float(speed))

    for player in PLAYERS:
        speeds = serve_speeds.get(player)
        if speeds:
            players[player]["serve"]["avg_speed_kmh"] = float(sum(speeds) / len(speeds))
            players[player]["serve"]["max_speed_kmh"] = float(max(speeds))

    # 4) 弹跳热图：全部回合的全部弹跳（不限定是否落在某一分内）。
    upper_half = _empty_grid()
    lower_half = _empty_grid()
    for rally in rallies:
        for bounce in rally.bounces:
            x, y = bounce["court"][0], bounce["court"][1]
            col = _clamp_index(x, GRID_COLS)
            if y < HALF_COURT_LENGTH_M:
                row = _clamp_index(y, GRID_ROWS)
                upper_half[row][col] += 1
            else:
                row = _clamp_index(y - HALF_COURT_LENGTH_M, GRID_ROWS)
                lower_half[row][col] += 1

    return {
        "players": players,
        "rally_length_histogram": histogram,
        "bounce_heatmap": {
            "grid_m": GRID_M,
            "upper_half": upper_half,
            "lower_half": lower_half,
        },
    }
