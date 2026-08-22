"""物理约束 3D 轨迹拟合：优化 (p0, v0, spin) 使仿真弹道的投影贴合 2D 观测。

残差 = 逐帧重投影像素误差 + 弹跳锚点误差（强权重，米->等效像素）
     + 击球点先验（弱权重，防止深度歧义漂移）。
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .physics import simulate_trajectory, sample_at

BOUNCE_WEIGHT_PX_PER_M = 400.0   # 弹跳点 1m 误差折合 400px 残差（强锚）
HINT_WEIGHT_PX_PER_M = 6.0       # 击球点先验弱约束

@dataclass
class FitResult:
    speed_kmh: float
    v0: list
    p0: list
    spin: float
    rms_px: float
    ok: bool

def fit_segment(camera, frame_times, ball_px, bounce_court_xy, hit_hint_xyz, max_rms=12.0):
    frame_times = np.asarray(frame_times, dtype=float)
    ball_px = np.asarray(ball_px, dtype=float)
    hint = np.asarray(hit_hint_xyz, dtype=float)
    valid = ~np.isnan(ball_px).any(axis=1) if ball_px.size else np.zeros(0, dtype=bool)

    # 退化输入守卫：帧数不足以定位轨迹（<2 帧）或有效观测不足 ok 判据的下限（<8）
    # 时直接判失败，不进 least_squares —— 否则 frame_times[-1] 越界，或初值猜测
    # （见下方 v_guess）在极短/零时长时发散到边界外触发 ValueError。
    if frame_times.size < 2 or int(valid.sum()) < 8:
        return FitResult(speed_kmh=0.0, v0=[0.0, 0.0, 0.0], p0=[float(x) for x in hint],
                         spin=0.0, rms_px=float("inf"), ok=False)

    duration = float(frame_times[-1]) + 0.02
    bounce = np.asarray(bounce_court_xy, dtype=float)

    def residuals(params):
        p0, v0, spin = params[0:3], params[3:6], params[6]
        times, pos = simulate_trajectory(p0, v0, spin, duration)
        sampled = sample_at(times, pos, frame_times)
        proj = camera.project(sampled)
        r_px = (proj[valid] - ball_px[valid]).ravel()
        # 末帧≈弹跳：位置贴合单应性坐标，高度贴地。
        # 注意：真实落地时刻可能比最后一个观测帧晚最多一个帧周期
        # （帧率越低这个 gap 越大：~6ms@60fps, ~23ms@30fps），因此该锚点
        # 存在与 1/fps 成比例的时间/位置偏差（实测 30fps 下 y 方向最多约
        # 0.47m）。该偏差由拟合在容差内吸收（30fps 下实测速度误差
        # 0.24%，仍远低于 8% 预算），低帧率 + 陡峭下落轨迹会侵蚀这一余量。
        # controller ruling R5：设计维持不变，此处仅补充说明，不改算法。
        end = sampled[-1]
        r_bounce = np.array([end[0] - bounce[0], end[1] - bounce[1], end[2] - 0.033]) * BOUNCE_WEIGHT_PX_PER_M
        r_hint = (p0 - hint) * HINT_WEIGHT_PX_PER_M
        return np.concatenate([r_px, r_bounce, r_hint])

    # 初值：p0=先验；v0 由「先验->弹跳点」直线飞行时间粗估；spin=0
    t_total = max(float(frame_times[-1]), 1e-3)
    v_guess = np.array([(bounce[0] - hint[0]) / t_total, (bounce[1] - hint[1]) / t_total, 1.0])
    x0 = np.concatenate([hint, v_guess, [0.0]])
    lb = [0 - 3, 0 - 3, 0.0, -80, -80, -30, -1.5]
    ub = [10.97 + 3, 23.77 + 3, 3.5, 80, 80, 30, 1.5]
    x0 = np.clip(x0, lb, ub)  # 防御性：极短/退化段的粗估 v_guess 可能越界，clip 保证 least_squares 初值合法
    sol = least_squares(residuals, x0, bounds=(lb, ub), x_scale=[1, 1, 1, 10, 10, 5, 0.3], max_nfev=200)

    p0, v0, spin = sol.x[0:3], sol.x[3:6], float(sol.x[6])
    times, pos = simulate_trajectory(p0, v0, spin, duration)
    proj = camera.project(sample_at(times, pos, frame_times))
    rms = float(np.sqrt(np.mean(np.sum((proj[valid] - ball_px[valid]) ** 2, axis=1))))
    speed = float(np.linalg.norm(v0) * 3.6)
    ok = bool(rms < max_rms and 10.0 < speed < 260.0 and int(valid.sum()) >= 8)
    return FitResult(speed_kmh=round(speed, 1), v0=[round(float(x), 2) for x in v0],
                     p0=[round(float(x), 2) for x in p0], spin=round(spin, 3),
                     rms_px=round(rms, 2), ok=ok)
