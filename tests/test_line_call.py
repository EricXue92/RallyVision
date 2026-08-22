import pytest

from tennis_analysis.analysis.line_call import call_bounce


def test_center_of_court_is_in():
    verdict, dist = call_bounce((5.485, 11.885), mode="doubles")
    assert verdict == "in"
    assert dist >= 0


def test_far_outside_doubles_alley_is_out():
    verdict, dist = call_bounce((12.0, 5.0), mode="doubles")
    assert verdict == "out"
    assert dist > 0


def test_singles_sideline_out_but_doubles_in():
    verdict_singles, _ = call_bounce((10.0, 5.0), mode="singles")
    assert verdict_singles == "out"

    verdict_doubles, _ = call_bounce((10.0, 5.0), mode="doubles")
    assert verdict_doubles == "in"


def test_just_inside_singles_sideline_is_close():
    # 1.37 是单打边线 x 坐标，3cm 内侧
    verdict, dist = call_bounce((1.40, 11.0), mode="singles")
    assert verdict == "close"
    assert dist == pytest.approx(0.03, abs=1e-6)


def test_just_beyond_baseline_within_ball_diameter_is_close():
    # 底线外 6cm，不足整球出界（球直径 6.6cm），落在 close 带内
    verdict, dist = call_bounce((5.485, 23.83), mode="doubles")
    assert verdict == "close"
    assert dist == pytest.approx(0.06, abs=1e-6)


def test_exactly_on_line_is_close():
    verdict, dist = call_bounce((0.0, 11.885), mode="doubles")
    assert verdict == "close"
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_far_outside_corner_is_out_with_euclidean_distance():
    verdict, dist = call_bounce((15.0, 30.0), mode="doubles")
    assert verdict == "out"
    # 最近点是场角 (10.97, 23.77)
    expected = ((15.0 - 10.97) ** 2 + (30.0 - 23.77) ** 2) ** 0.5
    assert dist == pytest.approx(expected, abs=1e-6)


def test_whole_ball_out_beyond_margin_is_out_not_close():
    # 距边线 1.03m，远超 close_margin，应判 out
    verdict, dist = call_bounce((12.0, 5.0), mode="doubles", close_margin_m=0.15)
    assert verdict == "out"
    assert dist == pytest.approx(1.03, abs=1e-6)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        call_bounce((5.0, 5.0), mode="mixed")
