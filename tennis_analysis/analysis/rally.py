"""回合切分：单遍逐帧扫描，从击球（shots）、弹跳（bounces）与球可见性
（visible）切出回合（Rally）。

算法（binding，见 task-3-brief）：
- 起点 = `shot_type == 'serve'` 的拍，或球连续不可见达到（按 fps 缩放的）
  BALL_LOST_FRAMES 帧后的第一拍，或整段视频的第一拍（业余录像常掐头，
  可能没有 serve）。
- 终点（三取一，最先命中者）：
  ① 球连续不可见达到 BALL_LOST_FRAMES 帧（'ball_lost'，end_frame=不可见段首帧）
  ② 某弹跳 line_call=='out' 且其后 OUT_NO_REPLY_FRAMES 帧内无任何击球（'out_no_reply'）
  ③ 同侧连续两次弹跳、中间无击球（'double_bounce'）
  视频结束强制收尾（'video_end'）。
- 回合内不足 1 拍的丢弃（本实现中回合总是随首拍一起创建，天然满足）。

BALL_LOST_FRAMES / OUT_NO_REPLY_FRAMES 是 60fps 基准，按传入的 fps 线性缩放
（30fps 素材减半）。
"""
from dataclasses import dataclass

BALL_LOST_FRAMES = 45
OUT_NO_REPLY_FRAMES = 30


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    shots: list          # shot_metrics 项引用（含 shot_type）
    bounces: list        # 本回合内的弹跳
    end_reason: str      # 'ball_lost' | 'out_no_reply' | 'double_bounce' | 'video_end'


def extract_rallies(shots, bounces, visible, fps) -> list:
    scale = float(fps) / 60.0
    ball_lost_frames = max(1, round(BALL_LOST_FRAMES * scale))
    out_no_reply_frames = max(1, round(OUT_NO_REPLY_FRAMES * scale))

    shots_by_frame = {}
    for shot in shots:
        shots_by_frame.setdefault(int(shot["hit_frame"]), []).append(shot)
    bounces_by_frame = {}
    for bounce in bounces:
        bounces_by_frame.setdefault(int(bounce["frame"]), []).append(bounce)

    last_event_frame = max(
        [len(visible) - 1] + list(shots_by_frame) + list(bounces_by_frame)
    )
    total_frames = max(len(visible), last_event_frame + 1)

    def is_visible(frame):
        return visible[frame] if frame < len(visible) else True

    rallies = []
    state = {"current": None, "pending_out_frame": None, "pending_bounce_side": None}
    invisible_run_start = None
    invisible_count = 0
    long_invisible_seen = False
    seen_first_shot = False

    def close_rally(end_frame, reason):
        current = state["current"]
        if current is not None and current["shots"]:
            rallies.append(
                Rally(
                    start_frame=current["start_frame"],
                    end_frame=end_frame,
                    shots=current["shots"],
                    bounces=current["bounces"],
                    end_reason=reason,
                )
            )
        state["current"] = None
        state["pending_out_frame"] = None
        state["pending_bounce_side"] = None

    for frame in range(total_frames):
        if is_visible(frame):
            invisible_run_start = None
            invisible_count = 0
        else:
            if invisible_run_start is None:
                invisible_run_start = frame
            invisible_count += 1
            if invisible_count == ball_lost_frames:
                long_invisible_seen = True
                if state["current"] is not None:
                    close_rally(invisible_run_start, "ball_lost")

        for bounce in bounces_by_frame.get(frame, []):
            if state["current"] is None:
                continue
            state["current"]["bounces"].append(bounce)
            if (
                state["pending_bounce_side"] is not None
                and bounce["side"] == state["pending_bounce_side"]
            ):
                close_rally(bounce["frame"], "double_bounce")
                continue
            state["pending_bounce_side"] = bounce["side"]
            if bounce["line_call"] == "out":
                state["pending_out_frame"] = bounce["frame"]

        for shot in shots_by_frame.get(frame, []):
            if state["current"] is None:
                starts_rally = (
                    shot["shot_type"] == "serve"
                    or long_invisible_seen
                    or not seen_first_shot
                )
                if starts_rally:
                    state["current"] = {"start_frame": frame, "shots": [shot], "bounces": []}
            else:
                state["current"]["shots"].append(shot)

            seen_first_shot = True
            long_invisible_seen = False
            state["pending_out_frame"] = None
            state["pending_bounce_side"] = None

        # 超时判定放在同帧的 bounce/shot 处理之后：若本帧恰好有回击拍（重置了
        # pending_out_frame),不应误判为超时无回击而收尾（曾在此漏掉同帧回击拍）。
        if (
            state["current"] is not None
            and state["pending_out_frame"] is not None
            and frame - state["pending_out_frame"] >= out_no_reply_frames
        ):
            close_rally(state["pending_out_frame"], "out_no_reply")

    if state["current"] is not None:
        close_rally(total_frames - 1, "video_end")

    return rallies
