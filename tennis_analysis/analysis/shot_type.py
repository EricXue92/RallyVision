"""规则法击球类型分类（Tennis-Vision 规则法，参考 89.4% 命中率）。

纯函数，无 cv2/torch 依赖，供 rally 分段与系统编排消费。

场地几何常量本地复制自 `tennis_analysis/court/mapper.py`（该模块顶部 `import cv2`，
若直接 import 会把 cv2 依赖带进这个应当保持纯净可测的模块，故不 import，只复制数值）。
"""

SHOT_TYPES = ("serve", "overhead", "volley", "forehand", "backhand", "unknown")

TENNIS_COURT_LENGTH = 23.77          # 米，同 court/mapper.py TENNIS_COURT_LENGTH
NET_Y = TENNIS_COURT_LENGTH / 2       # 11.885，网线球场坐标 y

NET_DISTANCE_VOLLEY_M = 3.0
BASELINE_MARGIN_M = 1.0


def classify_shot(ball_image, player_court, player_image, hands,
                   hitter, handedness, is_first_shot):
    court_y = player_court[1]

    # serve：回合第一拍 + 击球者在底线后（距网 > 10.9m）
    if is_first_shot and (
        court_y < BASELINE_MARGIN_M
        or court_y > TENNIS_COURT_LENGTH - BASELINE_MARGIN_M
    ):
        return "serve"

    left, right = hands.get("left"), hands.get("right")
    wrist_ys = [w[1] for w in (left, right) if w is not None]

    # overhead：非第一拍 + 球高过头顶（球 image y < 两腕点 image y 最小值）
    if not is_first_shot and wrist_ys and ball_image[1] < min(wrist_ys):
        return "overhead"

    # volley：非第一拍 + 击球者距网 < 3m
    if not is_first_shot and abs(court_y - NET_Y) < NET_DISTANCE_VOLLEY_M:
        return "volley"

    # forehand / backhand：需 player_image 判 side；缺则用双腕中点替代
    ref_point = player_image
    if ref_point is None:
        wrists = [w for w in (left, right) if w is not None]
        if wrists:
            ref_point = [
                sum(w[0] for w in wrists) / len(wrists),
                sum(w[1] for w in wrists) / len(wrists),
            ]

    if ref_point is None:
        return "unknown"

    ball_on_image_right = ball_image[0] > ref_point[0]
    player_side_right = ball_on_image_right if hitter == "lower" else not ball_on_image_right
    forehand = player_side_right == (handedness == "right")
    return "forehand" if forehand else "backhand"
