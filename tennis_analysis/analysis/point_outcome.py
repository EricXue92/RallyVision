"""每分判定：由回合（Rally）序列推断每一分的赢家/原因，含发球 fault 与
下一回合的配对（一发 fault 并入二发，连续两个 fault 判双误）。

优先级（见 task-4-brief）：
1. 回合仅 1 拍且为 serve、首弹跳 out → fault，不出分，并入下一回合（serve_number=2）；
   连续第二个 fault → double_fault，接发方得分。
2. 末弹跳 out → 末拍击球者失分，reason='out'。
3. 末弹跳 in 且其后无回击（'double_bounce' 或 'ball_lost' 且末弹跳 in）→
   末拍击球者得分，reason='winner'。
4. 其余（close 收尾、视频截断等）→ winner=末弹跳侧的对方，reason='unknown'
   （无弹跳时退化为末拍击球者的对方）——不静默丢分，交给状态机/编辑工具兜底。
"""


def _flip(side):
    return "lower" if side == "upper" else "upper"


def _is_fault(rally):
    return (
        len(rally.shots) == 1
        and rally.shots[0]["shot_type"] == "serve"
        and bool(rally.bounces)
        and rally.bounces[0]["line_call"] == "out"
    )


def _resolve_outcome(rally):
    last_hitter = rally.shots[-1]["hitter"]
    if not rally.bounces:
        return _flip(last_hitter), "unknown"

    last_bounce = rally.bounces[-1]
    if last_bounce["line_call"] == "out":
        return _flip(last_hitter), "out"
    if last_bounce["line_call"] == "in" and rally.end_reason in ("double_bounce", "ball_lost"):
        return last_hitter, "winner"
    return _flip(last_bounce["side"]), "unknown"


def infer_points(rallies, first_server) -> list:
    points = []
    pending_fault = None  # {"index", "start_frame", "hitter"}，等待与下一回合配对

    for index, rally in enumerate(rallies):
        fault = _is_fault(rally)

        if pending_fault is None:
            if fault:
                pending_fault = {
                    "index": index,
                    "start_frame": rally.start_frame,
                    "hitter": rally.shots[0]["hitter"],
                }
                continue
            winner, reason = _resolve_outcome(rally)
            points.append(
                {
                    "winner": winner,
                    "reason": reason,
                    "start_frame": rally.start_frame,
                    "end_frame": rally.end_frame,
                    "serve_number": 1,
                    "rally_indices": [index],
                }
            )
            continue

        # pending_fault 存在：本回合与其配对
        if fault:
            points.append(
                {
                    "winner": _flip(pending_fault["hitter"]),
                    "reason": "double_fault",
                    "start_frame": pending_fault["start_frame"],
                    "end_frame": rally.end_frame,
                    "serve_number": 2,
                    "rally_indices": [pending_fault["index"], index],
                }
            )
        else:
            winner, reason = _resolve_outcome(rally)
            points.append(
                {
                    "winner": winner,
                    "reason": reason,
                    "start_frame": pending_fault["start_frame"],
                    "end_frame": rally.end_frame,
                    "serve_number": 2,
                    "rally_indices": [pending_fault["index"], index],
                }
            )
        pending_fault = None

    # 末尾单个未配对的 fault：无下一回合可并入，消耗掉不出分（不 crash）。
    return points
