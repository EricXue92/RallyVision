import numpy as np
from tennis_analysis.analysis.physics import simulate_trajectory, GRAVITY

def test_free_fall_without_spin_matches_gravity():
    # 从 2m 静止落体（无水平速度时阻力/马格努斯都≈0），0.5s 后 z ≈ 2 - 0.5*g*0.25
    times, pos = simulate_trajectory(p0=[5.0, 12.0, 2.0], v0=[0.0, 0.0, 0.0], spin=0.0, duration=0.5)
    assert abs(pos[-1][2] - (2.0 - 0.5 * GRAVITY * 0.25)) < 0.01

def test_drag_reduces_range():
    # 有阻力的平抛射程必须小于真空抛物线
    times, pos = simulate_trajectory(p0=[5.0, 0.0, 1.0], v0=[0.0, 40.0, 0.0], spin=0.0, duration=0.4)
    vacuum_y = 40.0 * 0.4
    assert pos[-1][1] < vacuum_y - 0.3

def test_topspin_dips_below_flat():
    _, flat = simulate_trajectory(p0=[5.0, 0.0, 1.2], v0=[0.0, 30.0, 2.0], spin=0.0, duration=0.5)
    _, top = simulate_trajectory(p0=[5.0, 0.0, 1.2], v0=[0.0, 30.0, 2.0], spin=0.6, duration=0.5)
    assert top[-1][2] < flat[-1][2] - 0.05  # 上旋提前下坠

def test_slice_floats_above_flat():
    _, flat = simulate_trajectory(p0=[5.0, 0.0, 1.2], v0=[0.0, 30.0, 2.0], spin=0.0, duration=0.5)
    _, sl = simulate_trajectory(p0=[5.0, 0.0, 1.2], v0=[0.0, 30.0, 2.0], spin=-0.6, duration=0.5)
    assert sl[-1][2] > flat[-1][2] + 0.05
