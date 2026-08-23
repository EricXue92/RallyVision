from tennis_analysis.analysis.point_outcome import infer_points
from tennis_analysis.analysis.rally import Rally


def _serve_fault_rally(start_frame, hitter, bounce_side):
    # 单拍 serve，首弹跳 out → fault，不出分
    return Rally(
        start_frame=start_frame,
        end_frame=start_frame + 20,
        shots=[{"hit_frame": start_frame, "shot_type": "serve", "hitter": hitter}],
        bounces=[{"frame": start_frame + 15, "court": [1.0, 1.0], "line_call": "out", "side": bounce_side}],
        end_reason="out_no_reply",
    )


def test_single_fault_then_second_serve_point_won():
    # 回合 0：lower 一发 fault（发向 upper 半场界外）
    fault = _serve_fault_rally(0, "lower", "upper")
    # 回合 1：二发正常打完，upper 双误未接住，double_bounce 收尾，lower 得分
    second = Rally(
        start_frame=50,
        end_frame=120,
        shots=[
            {"hit_frame": 50, "shot_type": "serve", "hitter": "lower"},
            {"hit_frame": 80, "shot_type": "forehand", "hitter": "upper"},
        ],
        bounces=[
            {"frame": 100, "court": [5.0, 3.0], "line_call": "in", "side": "lower"},
            {"frame": 110, "court": [5.0, 3.5], "line_call": "in", "side": "lower"},
        ],
        end_reason="double_bounce",
    )
    points = infer_points([fault, second], first_server="lower")

    assert len(points) == 1
    point = points[0]
    assert point["serve_number"] == 2
    assert point["rally_indices"] == [0, 1]
    assert point["start_frame"] == fault.start_frame
    assert point["end_frame"] == second.end_frame
    assert point["winner"] == "upper"
    assert point["reason"] == "winner"


def test_double_fault_receiver_wins():
    fault1 = _serve_fault_rally(0, "lower", "upper")
    fault2 = _serve_fault_rally(50, "lower", "upper")

    points = infer_points([fault1, fault2], first_server="lower")

    assert len(points) == 1
    point = points[0]
    assert point["reason"] == "double_fault"
    assert point["winner"] == "upper"  # 接发方
    assert point["serve_number"] == 2
    assert point["rally_indices"] == [0, 1]
    assert point["start_frame"] == fault1.start_frame
    assert point["end_frame"] == fault2.end_frame


def test_last_bounce_out_hitter_loses():
    rally = Rally(
        start_frame=0,
        end_frame=60,
        shots=[
            {"hit_frame": 0, "shot_type": "serve", "hitter": "lower"},
            {"hit_frame": 30, "shot_type": "forehand", "hitter": "upper"},
        ],
        bounces=[
            {"frame": 50, "court": [5.0, 22.0], "line_call": "out", "side": "lower"},
        ],
        end_reason="out_no_reply",
    )

    points = infer_points([rally], first_server="lower")

    assert len(points) == 1
    point = points[0]
    assert point["reason"] == "out"
    assert point["winner"] == "lower"  # upper（最后击球者）失分，lower 得分
    assert point["serve_number"] == 1
    assert point["rally_indices"] == [0]
    assert point["start_frame"] == 0
    assert point["end_frame"] == 60


def test_double_bounce_ending_in_bounds_hitter_wins():
    rally = Rally(
        start_frame=0,
        end_frame=90,
        shots=[
            {"hit_frame": 0, "shot_type": "serve", "hitter": "lower"},
            {"hit_frame": 30, "shot_type": "forehand", "hitter": "upper"},
        ],
        bounces=[
            {"frame": 60, "court": [5.0, 21.0], "line_call": "in", "side": "lower"},
            {"frame": 70, "court": [5.0, 21.5], "line_call": "in", "side": "lower"},
        ],
        end_reason="double_bounce",
    )

    points = infer_points([rally], first_server="lower")

    assert len(points) == 1
    point = points[0]
    assert point["reason"] == "winner"
    assert point["winner"] == "upper"  # 最后击球者
    assert point["serve_number"] == 1


def test_close_call_ending_still_produces_unknown_point():
    rally = Rally(
        start_frame=0,
        end_frame=90,
        shots=[
            {"hit_frame": 0, "shot_type": "serve", "hitter": "lower"},
            {"hit_frame": 30, "shot_type": "forehand", "hitter": "upper"},
        ],
        bounces=[
            {"frame": 60, "court": [5.0, 21.9], "line_call": "close", "side": "lower"},
        ],
        end_reason="video_end",
    )

    points = infer_points([rally], first_server="lower")

    assert len(points) == 1
    point = points[0]
    assert point["reason"] == "unknown"
    assert point["winner"] == "upper"  # 末弹跳侧(lower)的对方
    assert point["serve_number"] == 1
    assert point["rally_indices"] == [0]
