"""比赛层编排（Task 7 binding）：把阶段 2 的 shot_metrics + detections.jsonl 逐帧
数据接成 shot_type -> rally -> point_outcome -> MatchState -> stats 的完整链路。

纯 dict/list 编排，无 cv2/torch 依赖，可脱离视频单测（system.py 只负责读盘/写盘 +
异常兜底，调用本模块的 `run_match_layer`，见 task-7-brief Step 2）。
"""
from .point_outcome import infer_points
from .rally import BALL_LOST_FRAMES, extract_rallies
from .scoring import MatchState
from .shot_type import NET_Y as HALF_COURT_LENGTH_M
from .shot_type import classify_shot
from .stats_report import build_stats

# scoring.MatchState 只认 0=upper/1=lower；这是 'upper'/'lower' 字符串与该数字
# 编码之间**唯一**的转换边界（brief 要求），其余模块（rally/point_outcome/
# stats_report）全程用字符串 side。
_SIDE_TO_INT = {"upper": 0, "lower": 1}

# 网线球场坐标 y（11.885）——直接从 shot_type.py 导入，不再本地重复硬编码
# （同一常量的第三处口径，见 stats_report.py HALF_COURT_LENGTH_M）。


def _handedness_for(hitter, upper_hand, lower_hand):
    return upper_hand if hitter == "upper" else lower_hand


def _hit_context(detections_by_frame, hit_frame, hitter):
    """从 detections.jsonl 逐帧记录里取该拍需要的 classify_shot 输入。找不到该帧
    记录、或该帧没有 hitter 一侧的球员数据时返回 None（调用方据此退化为 "unknown"，
    不让单拍数据缺失中断整条比赛层链路）。"""
    record = detections_by_frame.get(int(hit_frame))
    if record is None:
        return None
    player = (record.get("players") or {}).get(hitter) or {}
    ball = record.get("tennis_ball") or {}
    return {
        "ball_image": ball.get("image"),
        "player_court": player.get("court"),
        "player_image": player.get("image"),
        "hands": player.get("hands") or {"left": None, "right": None},
    }


def attach_shot_types(shot_metrics_entries, detections_by_frame, fps, upper_hand="right", lower_hand="right"):
    """给每条 shot_metrics 条目追加 "shot_type" 字段（只增字段，不改原有字段；
    返回新列表，不修改入参，顺序与入参一致）。

    is_first_shot 用 serve 启发式判定（controller ruling，见 task-7-brief）：分类
    阶段回合还没切出来（先要有 shot_type 才能切回合，鸡生蛋问题），这里用「整条
    shot_metrics 里的第一拍，或与上一拍 hit_frame 的间隔 >= 按 fps 缩放的
    rally.BALL_LOST_FRAMES」近似判定"是否回合首拍"——只要求粗略地"像发球"，
    足够喂给 classify_shot 的 serve 分支（该分支只看 is_first_shot + 底线位置）。
    """
    ordered = sorted(enumerate(shot_metrics_entries), key=lambda pair: int(pair[1]["hit_frame"]))
    threshold = max(1, round(BALL_LOST_FRAMES * float(fps) / 60.0))

    result = [dict(entry) for entry in shot_metrics_entries]
    prev_hit_frame = None
    for original_index, entry in ordered:
        hit_frame = int(entry["hit_frame"])
        is_first_shot = prev_hit_frame is None or (hit_frame - prev_hit_frame) >= threshold
        prev_hit_frame = hit_frame

        hitter = entry.get("hitter")
        context = _hit_context(detections_by_frame, hit_frame, hitter)
        shot_type = "unknown"
        if context is not None and context["ball_image"] is not None and context["player_court"] is not None:
            shot_type = classify_shot(
                context["ball_image"], context["player_court"], context["player_image"],
                context["hands"], hitter, _handedness_for(hitter, upper_hand, lower_hand),
                is_first_shot,
            )
        result[original_index]["shot_type"] = shot_type
    return result


def build_bounces(detections_by_frame):
    """从 detections.jsonl 的逐帧 "bounce" 字段构建 rally.extract_rallies 需要的
    bounces 列表（frame/court/line_call/side）。side 按落点 court y 是否小于半场长
    （网线）判 upper/lower，与 shot_type.py NET_Y / stats_report.py 半场分界同一
    常量口径。"""
    bounces = []
    for frame, record in sorted(detections_by_frame.items()):
        bounce = record.get("bounce")
        if not bounce or bounce.get("court") is None:
            continue
        court = bounce["court"]
        side = "upper" if court[1] < HALF_COURT_LENGTH_M else "lower"
        bounces.append({
            "frame": int(frame),
            "court": court,
            "line_call": bounce.get("line_call"),
            "side": side,
        })
    return bounces


def build_visible(detections_by_frame, total_frames=None):
    """按帧号建立 "tennis_ball.image != null" 的可见性列表，下标直接对应帧号
    （detections.jsonl / shot_metrics 的 frame 号本身是 1-indexed，不做 rebase）。

    detections.jsonl 只在 is_court=True 的帧写记录（见 system.py `_process_frame`），
    未被写入的帧号（非球场视角/回放插播等）在此按"不可见"处理——这与真实语义一致：
    那些帧本来就没有球跟踪数据。"""
    max_frame = max(detections_by_frame) if detections_by_frame else 0
    if total_frames is not None:
        max_frame = max(max_frame, int(total_frames))
    visible = [False] * (max_frame + 1)
    for frame, record in detections_by_frame.items():
        ball = record.get("tennis_ball") or {}
        visible[int(frame)] = ball.get("image") is not None
    return visible


def run_match_layer(shot_metrics_entries, detections_by_frame, fps, *,
                     first_server="lower", upper_hand="right", lower_hand="right",
                     sets_to_win=2, no_ad=False, total_frames=None):
    """跑完整比赛层链路：shot_type -> rally -> point_outcome -> MatchState -> stats。

    Returns dict:
        shot_metrics: 追加了 "shot_type" 的 shot_metrics 列表（顺序同入参）
        rallies: analysis.rally.Rally 列表（供 highlights 复用，未做 JSON 序列化）
        points: analysis.point_outcome.infer_points 输出
        match_score: MatchState.to_dict() + "score_timeline"（每分结束后的
            score_line）+ "points" + "final"（末态 score_line）
        match_stats: analysis.stats_report.build_stats 输出
        rally_score_lines: 与 rallies 等长的字符串列表——每个回合对应"该分开始前"
            （即 apply_point 之前）的比分快照，按 point["rally_indices"] 映射；
            供 --highlights 用（highlights.export_highlights 要的是"这一分开始时
            记分牌长什么样"，不是结束后），未落入任何分的回合（理论上不存在，
            infer_points 覆盖了每个非 fault 回合；末尾未配对的单个 fault 例外）
            填 ""。
    """
    shots_with_type = attach_shot_types(
        shot_metrics_entries, detections_by_frame, fps,
        upper_hand=upper_hand, lower_hand=lower_hand,
    )
    bounces = build_bounces(detections_by_frame)
    visible = build_visible(detections_by_frame, total_frames=total_frames)

    rallies = extract_rallies(shots_with_type, bounces, visible, fps)
    points = infer_points(rallies, first_server)

    match_state = MatchState(sets_to_win=sets_to_win, no_ad=no_ad, server=_SIDE_TO_INT[first_server])
    score_timeline = []
    rally_score_lines = [""] * len(rallies)
    for point in points:
        pre_point_score = match_state.score_line()
        try:
            match_state.apply_point(_SIDE_TO_INT[point["winner"]], point["reason"], rally_frames=point["rally_indices"])
        except ValueError:
            print(
                "提示：比赛已提前结束（分数已封顶），后续分数被忽略 / "
                "Notice: match already finished, ignoring the remaining points"
            )
            break
        # 只有真正喂进 MatchState 的分才落进 rally_score_lines/score_timeline——
        # 提前结束后的多余分（infer_points 仍会算出来，因为它对回合数据判定
        # 输赢，不知道比赛已经打完）必须被整体丢弃，否则 points/score_timeline
        # 长度、build_stats 的 points_won 汇总都会与"真实终局比分"对不上。
        for rally_index in point["rally_indices"]:
            if 0 <= rally_index < len(rally_score_lines):
                rally_score_lines[rally_index] = pre_point_score
        score_timeline.append(match_state.score_line())

    # 与上面同一条不变量：points 从这里开始就是"真正被计入终局比分的分"，
    # 下面的 match_score["points"] 和 build_stats 都必须用这个截断后的列表，
    # 不能用原始 infer_points 输出（那份可能比赛已结束后还有多余的分）。
    points = points[:len(score_timeline)]

    match_score = match_state.to_dict()
    match_score["score_timeline"] = score_timeline
    match_score["points"] = points
    match_score["final"] = match_state.score_line()

    match_stats = build_stats(rallies, points, shots_with_type)

    return {
        "shot_metrics": shots_with_type,
        "rallies": rallies,
        "points": points,
        "match_score": match_score,
        "match_stats": match_stats,
        "rally_score_lines": rally_score_lines,
    }
