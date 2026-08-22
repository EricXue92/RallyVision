"""三信号旋转分类器：轨迹曲率（trajectory）+ 弹跳恢复比（bounce）+ 挥拍腕部位移（swing）。

三个信号各自归一化到 [-1, 1]（正号=上旋方向），按权重加权求和后与阈值比较得出
label；confidence 由 |加权和| 与「信号一致度」相乘得到——加权和小可能是真的
"flat"，也可能是强信号互相抵消（矛盾），后者必须靠一致度把 confidence 压低，
否则下游会把矛盾样本当成高置信度的 flat 使用。

某一路信号缺失（wrist_dy=None，或 pre_bounce_vy≈0 导致恢复比不可算）时，直接把
该路从加权和与一致度计算中剔除，权重按比例分给其余信号——而不是把信号值当 0
带权参与，那样会把「不知道」悄悄计成「中性证据」，稀释真实信号的强度。
"""

# 各信号权重（无量纲，缺失信号的权重按比例分给其余信号）
W_TRAJECTORY = 0.5  # s1: FitResult.spin 曲率信号
W_BOUNCE = 0.3  # s2: 弹跳恢复比信号
W_SWING = 0.2  # s3: 挥拍腕部竖直位移信号

# s1：spin_coeff 归一化尺度（典型范围 [-1, 1]，除以此值后 clip）
SPIN_COEFF_SCALE = 0.35

# s2：弹跳恢复比 -post_bounce_vy/pre_bounce_vy 相对平击基准的偏差
BOUNCE_BASELINE_RATIO = 0.68  # 平击基准恢复比
BOUNCE_RATIO_SCALE = 0.35  # 偏差归一化尺度
PRE_BOUNCE_VY_EPS = 1e-6  # pre_bounce_vy 判零阈值，避免除零

# s3：腕部竖直位移归一化尺度（负值=由低向高=上旋）
WRIST_DY_SCALE = 0.3

# 加权和 -> label 的判定阈值
TOPSPIN_THRESHOLD = 0.25
SLICE_THRESHOLD = -0.25


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _trajectory_signal(spin_coeff: float) -> float:
    return _clip(spin_coeff / SPIN_COEFF_SCALE)


def _bounce_signal(pre_bounce_vy: float, post_bounce_vy: float):
    """返回 (signal, available)。pre_bounce_vy 近零时恢复比不可算，信号不可用。"""
    if abs(pre_bounce_vy) < PRE_BOUNCE_VY_EPS:
        return 0.0, False
    ratio = -post_bounce_vy / pre_bounce_vy
    return _clip((ratio - BOUNCE_BASELINE_RATIO) / BOUNCE_RATIO_SCALE), True


def _swing_signal(wrist_dy):
    """返回 (signal, available)。wrist_dy=None 表示无姿态数据，信号不可用。"""
    if wrist_dy is None:
        return 0.0, False
    return _clip(-(wrist_dy / WRIST_DY_SCALE)), True


def _signal_agreement(signals) -> float:
    """信号间两两符号一致度，映射到 [0, 1]；0/1 个可用信号时视为完全一致。"""
    if len(signals) < 2:
        return 1.0
    pair_scores = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            pair_scores.append(_sign(signals[i]) * _sign(signals[j]))
    mean_score = sum(pair_scores) / len(pair_scores)
    return (mean_score + 1.0) / 2.0


def classify_spin(spin_coeff, pre_bounce_vy, post_bounce_vy, wrist_dy=None):
    """三信号加权投票判定旋转类型。

    Args:
        spin_coeff: FitResult.spin，正=上旋，典型范围 [-1, 1]。
        pre_bounce_vy: 弹跳前 image 空间竖直速度（y-down 正）。
        post_bounce_vy: 弹跳后 image 空间竖直速度（弹跳后符号翻转）。
        wrist_dy: 击球者腕部竖直位移，负=由低向高（上旋）；None=无姿态数据。

    Returns:
        (label, confidence)：label ∈ {"topspin", "flat", "slice"}，
        confidence ∈ [0, 1]。
    """
    s1 = _trajectory_signal(spin_coeff)
    s2, bounce_available = _bounce_signal(pre_bounce_vy, post_bounce_vy)
    s3, swing_available = _swing_signal(wrist_dy)

    weighted_entries = [(s1, W_TRAJECTORY, True), (s2, W_BOUNCE, bounce_available), (s3, W_SWING, swing_available)]
    available = [(value, weight) for value, weight, is_available in weighted_entries if is_available]

    total_weight = sum(weight for _, weight in available)
    weighted_sum = sum(value * weight for value, weight in available) / total_weight

    if weighted_sum > TOPSPIN_THRESHOLD:
        label = "topspin"
    elif weighted_sum < SLICE_THRESHOLD:
        label = "slice"
    else:
        label = "flat"

    agreement = _signal_agreement([value for value, _ in available])
    confidence = _clip(abs(weighted_sum) * agreement)
    # abs(weighted_sum) 已 >=0，clip 只做上界保护，故 confidence 落在 [0, 1]
    confidence = max(0.0, confidence)

    return label, confidence
