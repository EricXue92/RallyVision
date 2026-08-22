import numpy as np
import pytest
from tennis_analysis.analysis.physics import simulate_trajectory, sample_at
from tennis_analysis.analysis.trajectory3d import fit_segment
from tests.test_camera import _lookat_extrinsics, W, H  # 复用测试相机搭建
from tennis_analysis.court.camera import CameraModel


def _make_camera():
    K = np.array([[1400.0, 0, W / 2], [0, 1400.0, H / 2], [0, 0, 1]])
    rvec, tvec = _lookat_extrinsics(np.array([5.485, -6.0, 3.0]), np.array([5.485, 11.885, 1.0]))
    return CameraModel(K, rvec, tvec)


def _synthesize(v0, spin, fps=60.0, noise_px=1.5, seed=7):
    cam = _make_camera()
    p0 = np.array([5.0, 2.5, 1.1])          # 底线附近 1.1m 高击球
    times, pos = simulate_trajectory(p0, v0, spin, duration=1.2)
    landing = np.argmax(pos[1:, 2] < 0.033) + 1   # 首次落到球半径高度
    t_land = times[landing]
    frame_times = np.arange(0, t_land, 1.0 / fps)
    pts3d = sample_at(times, pos, frame_times)
    px = cam.project(pts3d)
    rng = np.random.default_rng(seed)
    px = px + rng.normal(0, noise_px, px.shape)
    bounce_xy = pos[landing][:2]
    return cam, frame_times, px, bounce_xy, p0


def test_recovers_speed_within_5pct_flat():
    v0 = np.array([0.5, 28.0, 1.5])         # ~101 km/h 平击
    cam, ft, px, bxy, p0 = _synthesize(v0, spin=0.0)
    fit = fit_segment(cam, ft, px, bxy, hit_hint_xyz=[5.0, 2.5, 1.1])
    assert fit.ok
    truth = np.linalg.norm(v0) * 3.6
    assert abs(fit.speed_kmh - truth) / truth < 0.05


def test_recovers_topspin_sign():
    cam, ft, px, bxy, p0 = _synthesize(np.array([0.0, 26.0, 3.0]), spin=0.5)
    fit = fit_segment(cam, ft, px, bxy, hit_hint_xyz=[5.0, 2.5, 1.1])
    assert fit.ok and fit.spin > 0.15


def test_recovers_slice_sign():
    cam, ft, px, bxy, p0 = _synthesize(np.array([0.0, 22.0, 1.0]), spin=-0.5)
    fit = fit_segment(cam, ft, px, bxy, hit_hint_xyz=[5.0, 2.5, 1.1])
    assert fit.ok and fit.spin < -0.15


def test_missing_frames_tolerated():
    cam, ft, px, bxy, p0 = _synthesize(np.array([0.5, 28.0, 1.5]), spin=0.0)
    px[3:6] = np.nan                        # 连续 3 帧缺测
    fit = fit_segment(cam, ft, px, bxy, hit_hint_xyz=[5.0, 2.5, 1.1])
    assert fit.ok


@pytest.mark.parametrize("fps,max_err", [(60.0, 0.05), (30.0, 0.08)])
def test_recovers_speed_within_tolerance_by_fps(fps, max_err):
    """量化不同帧率下的拟合精度差距：60fps vs 30fps。"""
    v0 = np.array([0.5, 28.0, 1.5])         # ~101 km/h 平击
    cam, ft, px, bxy, p0 = _synthesize(v0, spin=0.0, fps=fps)
    fit = fit_segment(cam, ft, px, bxy, hit_hint_xyz=[5.0, 2.5, 1.1])
    assert fit.ok
    truth = np.linalg.norm(v0) * 3.6
    assert abs(fit.speed_kmh - truth) / truth < max_err
