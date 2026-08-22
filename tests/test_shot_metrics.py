import numpy as np

from tennis_analysis.analysis.segments import ShotSegment
from tennis_analysis.analysis.shot_metrics import compute_shot_metrics
from tests.test_trajectory3d import _synthesize


def _good_segment():
    cam, ft, px, bxy, p0 = _synthesize(np.array([0.5, 28.0, 1.5]), spin=0.0)
    segment = ShotSegment(
        hit_frame=0,
        bounce_frame=len(ft) - 1,
        hitter="lower",
        frame_times=list(ft),
        ball_px=[list(row) for row in px],
        bounce_court_xy=list(bxy),
        hit_hint_xyz=[5.0, 2.5, 1.1],
    )
    return segment, cam


def _bad_segment(n=10):
    return ShotSegment(
        hit_frame=100,
        bounce_frame=100 + n - 1,
        hitter="upper",
        frame_times=[i / 60.0 for i in range(n)],
        ball_px=[None] * n,
        bounce_court_xy=[5.0, 10.0],
        hit_hint_xyz=[5.0, 2.5, 1.1],
    )


def test_mixed_good_and_bad_segments_kept_with_none_speed():
    good, cam = _good_segment()
    bad = _bad_segment()

    results = compute_shot_metrics([good, bad], cam)

    assert len(results) == 2

    good_result = results[0]
    assert good_result["fit_ok"] is True
    assert good_result["speed_kmh"] is not None
    assert good_result["spin_coeff"] is not None
    assert good_result["hit_frame"] == 0
    assert good_result["bounce_frame"] == good.bounce_frame
    assert good_result["hitter"] == "lower"

    bad_result = results[1]
    assert bad_result["fit_ok"] is False
    assert bad_result["speed_kmh"] is None
    assert bad_result["spin_coeff"] is None
    assert bad_result["hit_frame"] == 100
    assert bad_result["bounce_frame"] == 109
    assert bad_result["hitter"] == "upper"


def test_fit_exception_is_caught_and_recorded(capsys):
    class ExplodingCamera:
        def project(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    good, _cam = _good_segment()

    results = compute_shot_metrics([good], ExplodingCamera())

    assert len(results) == 1
    assert results[0]["fit_ok"] is False
    assert results[0]["speed_kmh"] is None
    assert results[0]["spin_coeff"] is None

    captured = capsys.readouterr()
    assert "警告" in captured.out
    assert "Warning" in captured.out
