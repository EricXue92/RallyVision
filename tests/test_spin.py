from tennis_analysis.analysis.spin import classify_spin


def test_strong_topspin():
    label, conf = classify_spin(spin_coeff=0.5, pre_bounce_vy=30, post_bounce_vy=-26, wrist_dy=-0.4)
    assert label == "topspin" and conf > 0.6


def test_slice_by_trajectory_alone():
    label, _ = classify_spin(spin_coeff=-0.45, pre_bounce_vy=18, post_bounce_vy=-8, wrist_dy=None)
    assert label == "slice"


def test_flat_default():
    label, _ = classify_spin(spin_coeff=0.05, pre_bounce_vy=25, post_bounce_vy=-17, wrist_dy=0.0)
    assert label == "flat"


def test_conflicting_signals_low_confidence():
    _, conf = classify_spin(spin_coeff=0.4, pre_bounce_vy=20, post_bounce_vy=-8, wrist_dy=0.35)
    assert conf < 0.5


def test_wrist_dy_none_redistributes_weight_to_two_signals():
    # 无姿态数据时,仅靠轨迹+弹跳两信号也应给出确定 label,且权重全部落在两者上
    label, conf = classify_spin(spin_coeff=0.5, pre_bounce_vy=30, post_bounce_vy=-26, wrist_dy=None)
    assert label == "topspin"
    assert conf > 0.0


def test_pre_bounce_vy_zero_guard_no_crash():
    # pre_bounce_vy=0 时恢复比除零不可算,弹跳信号应被剔除而非抛异常,
    # 权重让给轨迹+挥拍信号,仍能给出确定判定
    label, conf = classify_spin(spin_coeff=0.5, pre_bounce_vy=0, post_bounce_vy=-26, wrist_dy=-0.4)
    assert label == "topspin"
    assert conf > 0.0


def test_all_signals_missing_except_trajectory_still_confident():
    # 弹跳信号(pre=0)与挥拍信号(None)同时不可用时,只剩轨迹一路,
    # 一致度按约定视为 1.0,confidence 应直接等于 |s1|
    label, conf = classify_spin(spin_coeff=0.5, pre_bounce_vy=0, post_bounce_vy=0, wrist_dy=None)
    assert label == "topspin"
    assert conf == 1.0
