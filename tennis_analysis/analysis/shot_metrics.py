"""段级拟合编排：把 segments.extract_segments 切出的 ShotSegment 逐个喂给
trajectory3d.fit_segment，汇总成下游 overlay 用的 dict 列表。

R2（binding）：ball_px 里 None -> NaN 的转换在这里做，fit_segment 只认 NaN。
fit_ok=False 的段保留（speed/spin 置 None），不静默丢弃——下游 overlay 要显示"—"。
fit_segment 抛异常时捕获记为 fit_ok=False 并打印双语警告，继续处理后续段。
"""
import numpy as np

from .trajectory3d import fit_segment


def compute_shot_metrics(segments, camera) -> list:
    results = []
    for segment in segments:
        ball_px = np.array(
            [[np.nan, np.nan] if p is None else p for p in segment.ball_px],
            dtype=float,
        )
        try:
            fit = fit_segment(
                camera,
                segment.frame_times,
                ball_px,
                segment.bounce_court_xy,
                segment.hit_hint_xyz,
            )
        except Exception as exc:  # noqa: BLE001 - 拟合失败不能中断整段处理
            print(
                f"警告：第 {segment.hit_frame}-{segment.bounce_frame} 帧拟合抛出异常，"
                f"已跳过该段 / Warning: fit_segment raised for frames "
                f"{segment.hit_frame}-{segment.bounce_frame}, skipping: {exc}"
            )
            results.append(
                {
                    "hit_frame": segment.hit_frame,
                    "bounce_frame": segment.bounce_frame,
                    "hitter": segment.hitter,
                    "speed_kmh": None,
                    "spin_coeff": None,
                    "fit_ok": False,
                    "rms_px": None,
                }
            )
            continue

        results.append(
            {
                "hit_frame": segment.hit_frame,
                "bounce_frame": segment.bounce_frame,
                "hitter": segment.hitter,
                "speed_kmh": fit.speed_kmh if fit.ok else None,
                "spin_coeff": fit.spin if fit.ok else None,
                "fit_ok": fit.ok,
                "rms_px": fit.rms_px,
            }
        )
    return results
