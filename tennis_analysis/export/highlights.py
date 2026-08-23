"""集锦导出：从回合（Rally）中挑选精彩片段，逐段切出并叠加比分板，
再用 concat demuxer 拼接成一条集锦视频。

Highlight reel export: select noteworthy rallies, cut each one as a clip
with a scoreboard overlay, then concatenate the clips into a single
highlight video using ffmpeg's concat demuxer.

只用 subprocess 调 ffmpeg（不引入 ffmpeg-python 依赖），不 import cv2/torch。
Only shells out to ffmpeg via subprocess (no ffmpeg-python dependency);
no cv2/torch imports in this module.
"""
import os
import shutil
import subprocess
import tempfile


def select_highlight_rallies(rallies, points, min_shots=8) -> list:
    """挑选回合：拍数 >= min_shots，或该回合产生的分 reason == 'winner'。

    Select rallies for the highlight reel: either the rally is long enough
    (shot count >= min_shots), or it is part of a point whose outcome
    reason is 'winner' (a clean winner, worth showing even if short).
    保持原始顺序返回 / returns the qualifying rallies in their original order.
    """
    winner_rally_indices = set()
    for point in points:
        if point.get("reason") == "winner":
            winner_rally_indices.update(point.get("rally_indices", []))

    selected = []
    for index, rally in enumerate(rallies):
        if len(rally.shots) >= min_shots or index in winner_rally_indices:
            selected.append(rally)
    return selected


def _escape_drawtext(text) -> str:
    """把比分文本转义后嵌入 ffmpeg drawtext 的 text='...' 引号写法。

    Escape score text for embedding inside ffmpeg drawtext's text='...'
    quoted syntax. 根据 ffmpeg 自己的 filtergraph 转义规则（av_get_token）：
    一旦进入单引号，内容按字面逐字复制（冒号、反斜杠都不再有特殊含义），
    唯一需要处理的是字面单引号本身——用「收尾引号、转义插入一个字面引号、
    重新开引号」的写法 `'\''`（与 shell 里常见的技巧同构，但这里是 ffmpeg
    自己的解析器语义，不是 shell）。
    Once inside single quotes, ffmpeg's own filtergraph parser copies
    content verbatim (colons and backslashes lose their special meaning);
    the only character that still needs escaping is a literal single
    quote, via the close-escape-reopen idiom `'\''`.
    """
    text = "" if text is None else str(text)
    return text.replace("'", "'\\''")


def _escape_concat_path(path) -> str:
    """concat demuxer 的 file '...' 一行同样是单引号语法，转义同上。"""
    return str(path).replace("'", "'\\''")


def _drawtext_available(ffmpeg_bin) -> bool:
    """探测这份 ffmpeg 是否编译了 drawtext（需要 libfreetype/fontconfig）。

    Probe whether this ffmpeg build has the drawtext filter compiled in
    (requires libfreetype/fontconfig). Some distro/Homebrew bottles ship
    without it, in which case `ffmpeg -h filter=drawtext` prints
    "Unknown filter 'drawtext'." instead of the filter's help text.
    探测失败（ffmpeg 本身跑不起来等）一律当作不可用，不让探测本身抛异常。
    Any failure while probing is treated as "unavailable" — never raises.
    """
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-h", "filter=drawtext"],
            capture_output=True, text=True,
        )
    except OSError:
        return False
    combined = (result.stdout or "") + (result.stderr or "")
    return "Unknown filter" not in combined


def export_highlights(video_path, rallies, score_lines, out_path, fps) -> bool:
    """把已挑选好的回合逐段切出（叠加比分板）并拼接为一条集锦视频。

    video_path: 原始比赛视频路径
    rallies: 要导出的回合列表（调用方已用 select_highlight_rallies 挑好）
    score_lines: 与 rallies 逐一对齐的比分字符串列表（每回合一条）；
                 比 rallies 短时，缺的位置按 "" 处理
    out_path: 集锦视频输出路径
    fps: 视频帧率，用于把 start_frame/end_frame 换算成秒

    ffmpeg 不存在或任一步 ffmpeg 调用非零退出 → 打印双语警告、返回 False，
    绝不让主流程崩溃。
    ffmpeg missing, or any ffmpeg invocation exits nonzero → print a
    bilingual warning and return False; never raises into the main flow.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print(
            "[集锦导出] 未检测到 ffmpeg，已跳过导出 / "
            "[highlights] ffmpeg not found on PATH, skipping highlight export."
        )
        return False

    if not rallies:
        print(
            "[集锦导出] 没有可导出的回合，已跳过导出 / "
            "[highlights] No rallies to export, skipping."
        )
        return False

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    drawtext_available = _drawtext_available(ffmpeg_bin)
    if not drawtext_available:
        print(
            "[集锦导出] 当前 ffmpeg 未编译 drawtext 滤镜，记分板叠加已跳过，"
            "仍会导出无叠加的集锦（如需记分板叠加，可 `brew reinstall ffmpeg` "
            "换一份带 drawtext 的完整版）/ "
            "[highlights] This ffmpeg build has no drawtext filter compiled "
            "in — scoreboard overlay skipped, exporting highlights without "
            "overlay (a full ffmpeg build, e.g. via `brew reinstall ffmpeg`, "
            "restores it)."
        )

    with tempfile.TemporaryDirectory(prefix="rallyvision_highlights_") as tmp_dir:
        segment_paths = []
        for index, rally in enumerate(rallies):
            score_text = score_lines[index] if index < len(score_lines) else ""
            start_sec = rally.start_frame / float(fps)
            end_sec = rally.end_frame / float(fps)
            segment_path = os.path.join(tmp_dir, f"seg_{index:04d}.mp4")

            cmd = [
                ffmpeg_bin, "-y",
                "-ss", f"{start_sec:.3f}",
                "-to", f"{end_sec:.3f}",
                "-i", str(video_path),
            ]
            if drawtext_available:
                drawtext_filter = (
                    "drawtext=text='{text}':x=24:y=24:fontsize=36:"
                    "fontcolor=white:box=1:boxcolor=black@0.5"
                ).format(text=_escape_drawtext(score_text))
                cmd += ["-vf", drawtext_filter]
            cmd.append(segment_path)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(
                    "[集锦导出] ffmpeg 切片失败，已跳过导出 / "
                    "[highlights] ffmpeg failed while cutting a segment, aborting export."
                )
                print(result.stderr[-2000:])
                return False
            segment_paths.append(segment_path)

        concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for segment_path in segment_paths:
                f.write(f"file '{_escape_concat_path(segment_path)}'\n")

        concat_cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            str(out_path),
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                "[集锦导出] ffmpeg 拼接失败，已跳过导出 / "
                "[highlights] ffmpeg failed while concatenating segments, aborting export."
            )
            print(result.stderr[-2000:])
            return False

    print(f"[集锦导出] 集锦已导出 / [highlights] Highlight reel exported: {out_path}")
    return True
