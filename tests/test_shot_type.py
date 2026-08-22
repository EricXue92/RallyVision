from tennis_analysis.analysis.shot_type import classify_shot

HANDS = {"left": [600.0, 400.0], "right": [700.0, 400.0]}

def test_first_shot_behind_baseline_is_serve():
    assert classify_shot([650, 100], [5.5, 24.5], [650, 450], HANDS,
                         "lower", "right", is_first_shot=True) == "serve"

def test_ball_above_wrists_is_overhead():
    assert classify_shot([650, 300], [5.5, 20.0], [650, 450], HANDS,
                         "lower", "right", is_first_shot=False) == "overhead"

def test_near_net_is_volley():
    assert classify_shot([650, 450], [5.5, 13.0], [650, 450], HANDS,
                         "lower", "right", is_first_shot=False) == "volley"

def test_lower_righty_ball_on_image_right_is_forehand():
    assert classify_shot([760, 450], [5.5, 20.0], [650, 450], HANDS,
                         "lower", "right", is_first_shot=False) == "forehand"

def test_upper_righty_ball_on_image_right_is_backhand():
    # upper 面对相机：image 右 = 本人左侧 → 右手持拍者是反手
    assert classify_shot([760, 450], [5.5, 3.0], [650, 450], HANDS,
                         "upper", "right", is_first_shot=False) == "backhand"

def test_missing_everything_is_unknown():
    assert classify_shot([760, 450], [5.5, 20.0], None, {"left": None, "right": None},
                         "lower", "right", is_first_shot=False) == "unknown"
