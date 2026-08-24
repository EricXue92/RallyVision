"""match_layer 回合中段假 serve 降级回归（2026-08-24 排查）。

attach_shot_types 的 is_first_shot 用「拍间隔 >= 按 fps 缩放的 BALL_LOST_FRAMES」
近似回合首拍，但正常对拉两拍之间的飞行时间（25fps 下 31-96 帧实测）普遍超过
该阈值（25fps 下仅 19 帧），导致回合中段只要击球者站在底线附近就被误标 serve
（真实 job 855db525 一个回合切出 3 个 serve）。回合切分之后 serve 只能是各回合
首拍，其余必须用 is_first_shot=False 重新分类。
"""
from tennis_analysis.analysis.match_layer import run_match_layer

FPS = 25.0


def _detections(hit_specs, bounce_specs, total_frames):
    """全帧球可见（不触发 ball_lost 切分）；击球帧带 hitter 球员数据，
    弹跳帧带 bounce 字段。"""
    records = {
        frame: {"tennis_ball": {"image": [400.0, 300.0]}}
        for frame in range(1, total_frames + 1)
    }
    for frame, hitter, court in hit_specs:
        records[frame]["players"] = {
            hitter: {
                "court": court,
                "image": [500.0, 500.0],
                "hands": {"left": None, "right": None},
            }
        }
        records[frame]["tennis_ball"] = {"image": [600.0, 500.0]}
    for frame, court, line_call in bounce_specs:
        records[frame]["bounce"] = {"court": court, "line_call": line_call}
    return records


def test_mid_rally_baseline_shots_are_not_serve():
    # 同一回合三拍，拍间隔 50 帧（> 25fps 阈值 19），击球者全在底线后；
    # 弹跳上下半场交替（不触发 double_bounce 收尾），回合由 video_end 收尾。
    hits = [
        (10, "lower", [4.0, 24.5]),
        (60, "upper", [5.0, -0.5]),
        (110, "lower", [4.5, 24.2]),
    ]
    bounces = [
        (30, [4.0, 5.0], "in"),
        (80, [4.0, 18.0], "in"),
        (130, [4.0, 6.0], "in"),
    ]
    shot_metrics = [{"hit_frame": frame, "hitter": hitter} for frame, hitter, _ in hits]
    detections = _detections(hits, bounces, total_frames=160)

    result = run_match_layer(shot_metrics, detections, FPS)

    types = [shot["shot_type"] for shot in result["shot_metrics"]]
    assert types[0] == "serve", types
    assert "serve" not in types[1:], types
    # 回合结构不因降级而变：仍是一个回合三拍
    assert len(result["rallies"]) == 1
    assert len(result["rallies"][0].shots) == 3
    # Rally.shots 与 shot_metrics 是同一批 dict，降级要同步可见
    rally_types = [shot["shot_type"] for shot in result["rallies"][0].shots]
    assert rally_types == types
