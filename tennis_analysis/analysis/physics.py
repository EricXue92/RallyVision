"""网球 3D 弹道物理模型：重力 + 空气阻力 + 马格努斯力。

世界坐标系：x=场宽[0,10.97] y=场长[0,23.77] z=竖直向上，单位米。
spin 是无量纲升力系数：正=上旋（向下压），负=切削（上飘），典型范围 [-1, 1]。
"""
import math
import numpy as np

BALL_MASS = 0.057      # kg
BALL_RADIUS = 0.033    # m
AIR_DENSITY = 1.21     # kg/m^3
DRAG_COEFF = 0.55
GRAVITY = 9.81
_BALL_AREA = math.pi * BALL_RADIUS ** 2
K_DRAG = 0.5 * AIR_DENSITY * DRAG_COEFF * _BALL_AREA / BALL_MASS   # a_drag = -K_DRAG*|v|*v
K_LIFT = 0.5 * AIR_DENSITY * _BALL_AREA / BALL_MASS                # a_lift = K_LIFT*spin*|v|^2 * dir

_Z = np.array([0.0, 0.0, 1.0])

def simulate_trajectory(p0, v0, spin, duration, dt=0.002):
    """半隐式欧拉积分，返回 (times[N], positions[N,3])，含 t=0。"""
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    steps = max(1, int(round(duration / dt)))
    times = np.empty(steps + 1)
    positions = np.empty((steps + 1, 3))
    times[0] = 0.0
    positions[0] = p
    for i in range(1, steps + 1):
        speed = float(np.linalg.norm(v))
        acc = np.array([0.0, 0.0, -GRAVITY])
        if speed > 1e-9:
            acc = acc - K_DRAG * speed * v
            vhat = v / speed
            axis = np.cross(_Z, vhat)          # 水平自转轴（垂直于运动方向）
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                # 上旋(spin>0): (axis × vhat) 对近水平运动 ≈ -z，即向下压
                acc = acc + K_LIFT * spin * speed * speed * np.cross(axis / norm, vhat)
        v = v + acc * dt
        p = p + v * dt
        times[i] = i * dt
        positions[i] = p
    return times, positions

def sample_at(times, positions, query_times):
    """按帧时间戳线性插值采样仿真轨迹，返回 [M,3]。"""
    q = np.asarray(query_times, dtype=float)
    return np.stack([np.interp(q, times, positions[:, k]) for k in range(3)], axis=1)
