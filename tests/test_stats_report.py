import pytest

from tennis_analysis.analysis.point_outcome import infer_points
from tennis_analysis.analysis.rally import Rally
from tennis_analysis.analysis.stats_report import build_stats


def _shot(hit_frame, shot_type, hitter, speed_kmh, fit_ok):
    return {
        "hit_frame": hit_frame,
        "shot_type": shot_type,
        "hitter": hitter,
        "speed_kmh": speed_kmh,
        "fit_ok": fit_ok,
    }


def _bounce(frame, court, line_call, side):
    return {"frame": frame, "court": court, "line_call": line_call, "side": side}


def _fixture_rallies():
    # R0: 一发在界内，upper 正拍 winner（双弹跳收尾）
    r0 = Rally(
        start_frame=0,
        end_frame=40,
        shots=[
            _shot(0, "serve", "lower", 150.0, True),
            _shot(30, "forehand", "upper", 80.0, True),
        ],
        bounces=[
            _bounce(10, [5.0, 3.0], "in", "lower"),
            _bounce(40, [5.0, 3.5], "in", "lower"),
        ],
        end_reason="double_bounce",
    )
    # R1+R2: lower 双误，upper（接发方）得分
    r1 = Rally(
        start_frame=100,
        end_frame=110,
        shots=[_shot(100, "serve", "lower", 140.0, True)],
        bounces=[_bounce(110, [1.0, 1.0], "out", "upper")],
        end_reason="out_no_reply",
    )
    r2 = Rally(
        start_frame=150,
        end_frame=160,
        shots=[_shot(150, "serve", "lower", 60.0, False)],  # fit_ok=False：不应进速度统计
        bounces=[_bounce(160, [1.0, 1.0], "out", "upper")],
        end_reason="out_no_reply",
    )
    # R3: 与 R0 同型，再给 upper 一次正拍 winner
    r3 = Rally(
        start_frame=300,
        end_frame=340,
        shots=[
            _shot(300, "serve", "lower", 145.0, True),
            _shot(330, "forehand", "upper", 85.0, True),
        ],
        bounces=[
            _bounce(310, [5.0, 3.0], "in", "lower"),
            _bounce(340, [5.0, 3.5], "in", "lower"),
        ],
        end_reason="double_bounce",
    )
    # R4: upper 正拍出界，lower 得分
    r4 = Rally(
        start_frame=400,
        end_frame=440,
        shots=[
            _shot(400, "serve", "lower", 148.0, True),
            _shot(430, "forehand", "upper", 90.0, True),
        ],
        bounces=[_bounce(440, [5.0, 22.5], "out", "lower")],
        end_reason="out_no_reply",
    )
    # R5: upper 发球，lower 反拍 winner（双弹跳收尾）
    r5 = Rally(
        start_frame=500,
        end_frame=540,
        shots=[
            _shot(500, "serve", "upper", 130.0, True),
            _shot(530, "backhand", "lower", 70.0, True),
        ],
        bounces=[
            _bounce(510, [5.0, 15.0], "in", "upper"),
            _bounce(540, [5.0, 15.5], "in", "upper"),
        ],
        end_reason="double_bounce",
    )
    # R6: upper 发球，lower 反拍出界，upper 得分；含一个出界坐标的弹跳（夹进边界格）
    r6 = Rally(
        start_frame=600,
        end_frame=640,
        shots=[
            _shot(600, "serve", "upper", 135.0, True),
            _shot(630, "backhand", "lower", 72.0, True),
        ],
        bounces=[
            _bounce(610, [-5.0, -3.0], "in", "upper"),  # 越界坐标，测试夹进边界格
            _bounce(640, [5.0, 23.5], "out", "upper"),
        ],
        end_reason="out_no_reply",
    )
    # R7: 10 拍长回合（serve + 4 组 forehand/volley 交替），upper 正拍 winner 收尾
    r7_shots = [_shot(700, "serve", "lower", 152.0, True)]
    for i in range(4):
        r7_shots.append(_shot(710 + i * 20, "forehand", "upper", 80.0 + i, True))
        r7_shots.append(_shot(720 + i * 20, "volley", "lower", 50.0 + i, True))
    r7_shots.append(_shot(790, "forehand", "upper", 84.0, True))
    r7 = Rally(
        start_frame=700,
        end_frame=798,
        shots=r7_shots,
        bounces=[
            _bounce(795, [5.0, 15.0], "in", "upper"),
            _bounce(798, [5.0, 15.5], "in", "upper"),
        ],
        end_reason="double_bounce",
    )
    return [r0, r1, r2, r3, r4, r5, r6, r7]


@pytest.fixture
def fixture():
    rallies = _fixture_rallies()
    points = infer_points(rallies, first_server="lower")
    shots = [shot for rally in rallies for shot in rally.shots]
    return rallies, points, shots


def test_points_are_shaped_as_expected(fixture):
    rallies, points, shots = fixture
    assert len(rallies) == 8
    assert len(points) == 7
    winners = [point["winner"] for point in points]
    assert winners == ["upper", "upper", "upper", "lower", "lower", "upper", "upper"]


def test_forehand_shot_count_and_point_win_rate(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    forehand = stats["players"]["upper"]["shots"]["forehand"]
    # R0/R3/R4 各一拍 + R7 五拍（710/730/750/770/790）
    assert forehand["count"] == 8
    # 样本分：point0(win)/point2(win)/point3(loss)/point6(win) -> 3/4
    assert forehand["point_win_rate"] == pytest.approx(0.75)


def test_point_win_rate_none_below_sample_threshold(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    # volley 只出现在 R7 对应的单个分里，样本 1 < 3 -> None
    volley = stats["players"]["lower"]["shots"]["volley"]
    assert volley["count"] == 4
    assert volley["point_win_rate"] is None


def test_double_faults_attributed_to_server(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    assert stats["players"]["lower"]["serve"]["double_faults"] == 1
    assert stats["players"]["upper"]["serve"]["double_faults"] == 0


def test_first_in_pct(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    # lower: 一发点(point0)+一发fault(point1)+一发点(point2)+一发点(point3)+一发点(point6，R7) = 4 in / 5 attempts
    assert stats["players"]["lower"]["serve"]["first_in_pct"] == pytest.approx(0.8)
    # upper: point4/point5 均一发在界
    assert stats["players"]["upper"]["serve"]["first_in_pct"] == pytest.approx(1.0)


def test_serve_speed_excludes_fit_ok_false(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    serve = stats["players"]["lower"]["serve"]
    # 60 km/h 的一拍 fit_ok=False，应被排除；否则均值会偏低到 132.5
    assert serve["avg_speed_kmh"] == pytest.approx(147.0)
    assert serve["max_speed_kmh"] == pytest.approx(152.0)


def test_points_won(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    assert stats["players"]["upper"]["points_won"] == 5
    assert stats["players"]["lower"]["points_won"] == 2


def test_rally_length_histogram_buckets(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    histogram = stats["rally_length_histogram"]
    assert histogram == {"1-3": 6, "4-6": 0, "7-9": 0, "10+": 1}


def test_bounce_heatmap_grid_sum_matches_total_bounces(fixture):
    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    heatmap = stats["bounce_heatmap"]
    assert heatmap["grid_m"] == 1.0
    assert len(heatmap["upper_half"]) == 12
    assert len(heatmap["upper_half"][0]) == 11
    assert len(heatmap["lower_half"]) == 12
    assert len(heatmap["lower_half"][0]) == 11

    total_bounces = sum(len(rally.bounces) for rally in rallies)
    grid_total = sum(sum(row) for row in heatmap["upper_half"]) + sum(
        sum(row) for row in heatmap["lower_half"]
    )
    assert grid_total == total_bounces == 13

    # 越界坐标 [-5.0, -3.0] 应被夹进 upper_half 的 [0][0] 格
    assert heatmap["upper_half"][0][0] == 1


def test_output_is_json_serializable(fixture):
    import json

    rallies, points, shots = fixture
    stats = build_stats(rallies, points, shots)
    # 不应有 numpy 标量泄漏；json.dumps 若含 numpy 类型会抛 TypeError
    json.dumps(stats)
