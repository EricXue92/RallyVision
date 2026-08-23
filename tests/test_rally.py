from tennis_analysis.analysis.rally import extract_rallies


def _visible_with_gap(length, gap_start, gap_end):
    """长度为 length 的可见性序列，[gap_start, gap_end) 之间球不可见，其余可见。"""
    v = [True] * length
    for f in range(gap_start, gap_end):
        v[f] = False
    return v


def _two_rally_fixture():
    # 回合 A：serve@10 -> 击球@60 -> 击球@110 -> out 弹跳@150 后无击球（out_no_reply）
    # 空档：f150-f400 球先可见后（f200-f390）不可见，均不属于任何回合
    # 回合 B：serve@400（此前球连续不可见 190 帧）-> in 弹跳@440(upper) -> in 弹跳@470(upper) 中间无击球（double_bounce）
    shots = [
        {"hit_frame": 10, "shot_type": "serve", "hitter": "lower"},
        {"hit_frame": 60, "shot_type": "forehand", "hitter": "upper"},
        {"hit_frame": 110, "shot_type": "backhand", "hitter": "lower"},
        {"hit_frame": 400, "shot_type": "serve", "hitter": "lower"},
    ]
    bounces = [
        {"frame": 150, "court": [5.5, 4.0], "line_call": "out", "side": "upper"},
        {"frame": 440, "court": [5.5, 3.0], "line_call": "in", "side": "upper"},
        {"frame": 470, "court": [5.5, 2.5], "line_call": "in", "side": "upper"},
    ]
    visible = _visible_with_gap(500, 200, 390)
    return shots, bounces, visible


def test_two_rallies_out_no_reply_and_double_bounce():
    shots, bounces, visible = _two_rally_fixture()
    rallies = extract_rallies(shots, bounces, visible, fps=60)

    assert len(rallies) == 2
    rally_a, rally_b = rallies

    assert rally_a.end_reason == "out_no_reply"
    assert len(rally_a.shots) == 3

    assert rally_b.end_reason == "double_bounce"
    assert rally_b.start_frame == 400

    # f150-f400 的空档不属于任何回合
    assert rally_a.end_frame < 400
    assert rally_b.start_frame >= 400
    for rally in rallies:
        for shot in rally.shots:
            assert not (150 < shot["hit_frame"] < 400)
        for bounce in rally.bounces:
            assert not (150 < bounce["frame"] < 400)


def test_rally_a_bounce_is_the_out_bounce():
    shots, bounces, visible = _two_rally_fixture()
    rallies = extract_rallies(shots, bounces, visible, fps=60)
    rally_a = rallies[0]
    assert len(rally_a.bounces) == 1
    assert rally_a.bounces[0]["line_call"] == "out"


def test_first_shot_opens_rally_even_without_serve():
    # 业余录像常掐头：全程球可见但没有任何 serve 标签，仍应在首拍开回合
    shots = [
        {"hit_frame": 5, "shot_type": "forehand", "hitter": "lower"},
        {"hit_frame": 55, "shot_type": "backhand", "hitter": "upper"},
        {"hit_frame": 105, "shot_type": "forehand", "hitter": "lower"},
    ]
    bounces = []
    visible = [True] * 150

    rallies = extract_rallies(shots, bounces, visible, fps=60)

    assert len(rallies) == 1
    rally = rallies[0]
    assert rally.start_frame == 5
    assert rally.end_reason == "video_end"
    assert len(rally.shots) == 3


def test_ball_lost_ends_rally():
    # 回合中球连续不可见 45 帧应立即收尾，end_frame=不可见段首帧
    shots = [
        {"hit_frame": 10, "shot_type": "serve", "hitter": "lower"},
        {"hit_frame": 60, "shot_type": "forehand", "hitter": "upper"},
    ]
    bounces = []
    visible = _visible_with_gap(200, 100, 100 + 45)

    rallies = extract_rallies(shots, bounces, visible, fps=60)

    assert len(rallies) == 1
    rally = rallies[0]
    assert rally.end_reason == "ball_lost"
    assert rally.end_frame == 100
    assert len(rally.shots) == 2


def test_reply_shot_exactly_at_timeout_boundary_is_not_dropped():
    # 回归测试：out 弹跳后恰好第 OUT_NO_REPLY_FRAMES 帧有回击拍，
    # 该拍必须被收进回合、回合不应被判定为 out_no_reply 提前收尾。
    # （曾因超时判定先于同帧的击球处理执行，导致这拍被静默丢弃。）
    shots = [
        {"hit_frame": 10, "shot_type": "serve", "hitter": "lower"},
        {"hit_frame": 50 + 30, "shot_type": "forehand", "hitter": "upper"},
    ]
    bounces = [
        {"frame": 50, "court": [5.5, 4.0], "line_call": "out", "side": "upper"},
    ]
    visible = [True] * 150

    rallies = extract_rallies(shots, bounces, visible, fps=60)

    assert len(rallies) == 1
    rally = rallies[0]
    assert rally.end_reason == "video_end"
    assert len(rally.shots) == 2
    assert any(shot["hit_frame"] == 80 for shot in rally.shots)
