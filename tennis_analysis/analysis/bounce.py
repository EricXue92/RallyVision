import bisect
import json
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..media.video_audio import encode_vscode_compatible_mp4
from ..visualization.minimap import MiniMapVisualizer
from ..visualization.stats import TextPatchRenderer


class LegacyColumnConcatenator(BaseEstimator, TransformerMixin):
    """Compatibility shim for old sktime ColumnConcatenator pickles."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rows = []
        for _, row in X.iterrows():
            values = []
            for value in row:
                if isinstance(value, pd.Series):
                    values.extend(value.tolist())
                elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    values.extend(list(value))
                else:
                    values.append(value)
            rows.append(pd.Series(values))
        return pd.DataFrame({"ts": rows})


@dataclass
class TrajectoryPoint:
    frame: int
    time_sec: float
    image: Optional[list]
    court: Optional[list]
    interpolated: bool = False


class BounceDetector:
    """Post-process ball trajectory with a 20-frame x/y/velocity window."""

    def __init__(
        self,
        fps=30,
        window_size=20,
        center_offset=10,
        min_event_gap_sec=0.5,
        min_score=0.34,
        max_interpolation_gap=12,
        classifier_path="",
        court_margin=0.9,
        max_center_velocity=2500,
        max_speed_ratio=12,
    ):
        self.fps = max(float(fps), 1.0)
        self.window_size = int(window_size)
        self.center_offset = int(center_offset)
        self.min_event_gap_frames = max(1, int(float(min_event_gap_sec) * self.fps))
        self.min_score = float(min_score)
        self.max_interpolation_gap = int(max_interpolation_gap)
        self.classifier_path = classifier_path
        self.court_margin = float(court_margin)
        self.max_center_velocity = float(max_center_velocity)
        self.max_speed_ratio = float(max_speed_ratio)
        self.classifier = None
        self.classifier_error = None
        self._classifier_kind = None  # 'catboost' | 'sktime'
        self.events = []
        self.processed_points = []

    def process_detections(self, detections_path, output_path=None, trajectory_output_path=None, rewrite_detections=True):
        records = self._load_records(detections_path)
        points = self._records_to_points(records)
        cleaned = self._remove_outliers(points)
        interpolated = self._interpolate(cleaned)
        self.processed_points = interpolated
        events = self.detect_from_points(interpolated)
        self.events = events

        if output_path:
            self._write_events(output_path, events)
        if trajectory_output_path:
            self._write_trajectory(trajectory_output_path, interpolated)
        if rewrite_detections:
            self._rewrite_records_with_bounces(detections_path, records, events)
        return events

    def detect_from_points(self, points):
        coords = np.array(
            [
                point.image if point.image is not None else [np.nan, np.nan]
                for point in points
            ],
            dtype=np.float32,
        )
        velocity = self._velocity(coords)
        raw_events = []
        classifier = self._load_classifier()
        if classifier is not None and self._classifier_kind == "catboost":
            # CatBoost 路径(移植自 yastrebksv/TennisProject):模型在带标注真实
            # 数据上训练,自带 bounce/hit 区分,不走规则链的吸附+否决(那套
            # 窗口启发式在「落点后 0.2s 内被回击」的常见场景会误杀真落点)。
            events = self._detect_with_catboost(classifier, points, coords)
            return self._finalize_events(points, events)
        if classifier is not None:
            events = self._detect_with_classifier(classifier, points, coords, velocity)
            return self._refine_events(points, events)

        court_coords = np.array(
            [
                point.court if point.court is not None else [np.nan, np.nan]
                for point in points
            ],
            dtype=np.float32,
        )

        for end_index in range(self.window_size - 1, len(points)):
            start_index = end_index - self.window_size + 1
            center_index = end_index - self.center_offset
            if center_index <= 0 or center_index >= len(points) - 1:
                continue

            window = coords[start_index:end_index + 1]
            window_v = velocity[start_index:end_index + 1]
            if np.isnan(window).any() or np.isnan(window_v).any():
                continue

            court_window = court_coords[start_index:end_index + 1]
            if np.isnan(court_window).any():
                court_window = None
            score, diagnostics = self._score_window(window, window_v, self.window_size - self.center_offset - 1, court_window=court_window)
            is_bounce = score >= self.min_score
            if not is_bounce:
                continue

            point = points[center_index]
            if not self._valid_bounce_court_position(point.court):
                continue
            raw_events.append(
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)],
                    "court": point.court,
                    "confidence": round(float(score), 3),
                    "method": "clf_lag20" if classifier is not None else "trajectory_lag20",
                    "diagnostics": diagnostics,
                }
            )

        # 去重在 _refine_events 链内做(吸附+击球点丢弃之后),这里传原始候选
        return self._refine_events(points, raw_events)

    # CatBoost 回归输出过阈值即候选;阈值与参考实现(yastrebksv/TennisProject)一致
    CATBOOST_THRESHOLD = 0.45

    def _detect_with_catboost(self, model, points, coords):
        """特征工程逐列复刻参考实现:±2 帧滞后的 x/y 差分与差分比,共 12 列。
        行按帧序对齐,检测缺失帧特征为 NaN、整行丢弃(与参考的 notna 过滤等价)。"""
        frame_series = pd.DataFrame(
            {
                "x": [float(c[0]) if np.isfinite(c[0]) else np.nan for c in coords],
                "y": [float(c[1]) if np.isfinite(c[1]) else np.nan for c in coords],
            }
        )
        num = 3
        eps = 1e-15
        for i in range(1, num):
            for col in ("x", "y"):
                frame_series[f"{col}_lag_{i}"] = frame_series[col].shift(i)
                frame_series[f"{col}_lag_inv_{i}"] = frame_series[col].shift(-i)
            frame_series[f"x_diff_{i}"] = (frame_series[f"x_lag_{i}"] - frame_series["x"]).abs()
            frame_series[f"y_diff_{i}"] = frame_series[f"y_lag_{i}"] - frame_series["y"]
            frame_series[f"x_diff_inv_{i}"] = (frame_series[f"x_lag_inv_{i}"] - frame_series["x"]).abs()
            frame_series[f"y_diff_inv_{i}"] = frame_series[f"y_lag_inv_{i}"] - frame_series["y"]
            frame_series[f"x_div_{i}"] = (frame_series[f"x_diff_{i}"] / (frame_series[f"x_diff_inv_{i}"] + eps)).abs()
            frame_series[f"y_div_{i}"] = frame_series[f"y_diff_{i}"] / (frame_series[f"y_diff_inv_{i}"] + eps)

        columns = (
            [f"x_diff_{i}" for i in range(1, num)]
            + [f"x_diff_inv_{i}" for i in range(1, num)]
            + [f"x_div_{i}" for i in range(1, num)]
            + [f"y_diff_{i}" for i in range(1, num)]
            + [f"y_diff_inv_{i}" for i in range(1, num)]
            + [f"y_div_{i}" for i in range(1, num)]
        )
        valid = frame_series[columns].notna().all(axis=1)
        features = frame_series.loc[valid, columns]
        if features.empty:
            return []
        predictions = model.predict(features)
        row_indices = list(features.index)

        raw_events = []
        for row_number, prob in enumerate(predictions):
            if prob <= self.CATBOOST_THRESHOLD:
                continue
            index = row_indices[row_number]
            # 参考实现的连续帧合并:相邻帧都过阈值时只留概率更高的一帧
            if raw_events and index - raw_events[-1][0] == 1:
                if prob > raw_events[-1][1]:
                    raw_events[-1] = (index, float(prob))
                continue
            raw_events.append((index, float(prob)))

        events = []
        for index, prob in raw_events:
            point = points[index]
            if point.image is None:
                continue
            if not self._valid_bounce_court_position(point.court):
                continue
            events.append(
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)],
                    "court": point.court,
                    "confidence": round(min(float(prob), 1.0), 3),
                    "method": "catboost_lag2",
                    "diagnostics": {
                        "classifier_path": self.classifier_path,
                        "raw_prediction": round(float(prob), 4),
                        "threshold": self.CATBOOST_THRESHOLD,
                    },
                }
            )
        return events

    # 边缘分候选的触地旁证:CatBoost 原始输出低于此值时,要求 ±该半径帧内
    # 存在图像 y 内部局部峰(屏幕最低点=触地)。已核数据:深度反转/局部峰值
    # 等逐帧硬否决在本管线全不可用(事件帧有 ±4 帧时间松散 + 空中投影伪影),
    # 只有「模型自己不确定时才要旁证」这一档能把 job 7bb0934f 的 f188 击球
    # 误报(0.451)和 f172 真落点(0.459)分开,不要把它改成对全量候选的否决
    # (f99 真落点 0.68 无峰,全量否决会误杀)。
    CATBOOST_CONFIDENT = 0.60
    TOUCHDOWN_PEAK_RADIUS = 3

    def _finalize_events(self, points, events):
        """CatBoost 路径的收尾:边缘分触地旁证(去重**之前**,否则高分击球
        误报先吃掉相邻真落点再被丢,两头落空)+ 去重(0.5s 窗口留最高置信,
        真落点会吃掉紧邻的击球残余候选)+ segments.refine_bounce 亚帧插值
        court 坐标。不做规则链的 y-max 吸附与深度反转否决。"""
        if not events:
            return events
        from .segments import refine_bounce

        dict_points = [
            {
                "frame": int(point.frame),
                "time_sec": float(point.time_sec),
                "image": list(point.image) if point.image is not None else None,
                "court": list(point.court) if point.court is not None else None,
            }
            for point in points
        ]
        image_y_by_frame = {
            p["frame"]: p["image"][1] for p in dict_points if p["image"] is not None
        }
        kept = [
            event for event in events
            if event["diagnostics"]["raw_prediction"] >= self.CATBOOST_CONFIDENT
            or self._touchdown_peak_near(image_y_by_frame, int(event["frame"]))
        ]
        refined_events = []
        for event in self._dedupe_events(kept):
            try:
                _, refined_court = refine_bounce(dict_points, int(event["frame"]))
            except (ValueError, TypeError):
                refined_court = None
            if refined_court is not None:
                diagnostics = event.setdefault("diagnostics", {})
                diagnostics["court_raw"] = event.get("court")
                event["court"] = [round(float(refined_court[0]), 2), round(float(refined_court[1]), 2)]
            refined_events.append(event)
        return refined_events

    def _touchdown_peak_near(self, image_y_by_frame, frame):
        """±TOUCHDOWN_PEAK_RADIUS 帧内是否存在图像 y 内部局部峰(两侧相邻帧都
        不高于它)。轨迹缺帧评不出峰时放行(fail open,不因数据洞杀真落点)。"""
        evaluable = False
        for f in range(frame - self.TOUCHDOWN_PEAK_RADIUS, frame + self.TOUCHDOWN_PEAK_RADIUS + 1):
            if f not in image_y_by_frame or (f - 1) not in image_y_by_frame or (f + 1) not in image_y_by_frame:
                continue
            evaluable = True
            y = image_y_by_frame[f]
            if y >= image_y_by_frame[f - 1] and y >= image_y_by_frame[f + 1]:
                return True
        return not evaluable

    def _detect_with_classifier(self, classifier, points, coords, velocity):
        feature_rows = []
        center_indices = []
        for row_index in range(self.window_size, len(points) - 1):
            window = coords[row_index - self.window_size:row_index]
            window_v = velocity[row_index - self.window_size:row_index]
            if np.isnan(window).any() or np.isnan(window_v).any():
                continue
            feature_rows.append(self._window_to_feature_row(window, window_v))
            center_indices.append(row_index - self.center_offset)

        if not feature_rows:
            return []

        features = pd.DataFrame(feature_rows)
        predictions = classifier.predict(features)
        probabilities = None
        if hasattr(classifier, "predict_proba"):
            try:
                probabilities = classifier.predict_proba(features)
            except Exception:
                probabilities = None

        classes = list(getattr(classifier, "classes_", []))
        raw_events = []
        for row_number, prediction in enumerate(predictions):
            if int(prediction) != 1:
                continue
            center_index = center_indices[row_number]
            if center_index < 0 or center_index >= len(points):
                continue
            point = points[center_index]
            if point.image is None:
                continue
            confidence = 1.0
            if probabilities is not None:
                if 1 in classes:
                    confidence = float(probabilities[row_number][classes.index(1)])
                elif len(probabilities[row_number]) > 1:
                    confidence = float(probabilities[row_number][-1])
            raw_events.append(
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)],
                    "court": point.court,
                    "confidence": round(float(confidence), 3),
                    "method": "clf_lag20",
                    "diagnostics": {
                        "classifier_path": self.classifier_path,
                        "prediction": int(prediction),
                        "window_size": int(self.window_size),
                        "lag_order": "20_to_1",
                    },
                }
            )
        # 去重在 _refine_events 链内做(吸附+击球点丢弃之后),这里返回原始候选
        return raw_events

    def annotate_video(
        self,
        input_video_path,
        output_video_path,
        events,
        trajectory_points=None,
        display_sec=0.45,
        trajectory_length=30,
        draw_minimap_bounces=True,
        draw_processed_trajectory=True,
        bounce_line_calls=None,
        shot_hits=None,
        rally_spans=None,
        rally_label_pos=None,
        rally_label_font_size=None,
    ):
        """Task 10: 追加 bounce_line_calls（frame -> "in"/"out"/"close"）在弹跳标记旁画判罚，
        shot_hits（hit_frame -> shot_metrics 条目 dict）在击球帧起 1.5s 画 "hitter speed · spin"。
        两者都是可选的（None/空 dict 时行为与 Task 10 之前完全一致）。

        rally_spans（[(start_frame, end_frame), ...] 升序）：真回合切分（extract_rallies
        口径）。传入时逐帧画「回合: N」——N = 已开始的回合数（当前帧之前/之中最后一个
        start_frame 的序号），位置/字号由 rally_label_pos / rally_label_font_size 给
        （几何单一来源在 StatsVisualizer.rally_label_geometry，第一遍已不画回合数）。
        """
        if not events and not trajectory_points:
            return False

        events = sorted(events, key=lambda event: int(event["frame"]))
        trajectory_points = trajectory_points or self.processed_points
        trajectory_by_frame = self._trajectory_by_frame(trajectory_points)
        bounce_line_calls = bounce_line_calls or {}
        shot_hits = shot_hits or {}
        video = cv2.VideoCapture(input_video_path)
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video for bounce annotation: {input_video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_output_video_path = f"{output_video_path}.raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            video.release()
            raise RuntimeError(f"Unable to create bounce annotation video: {raw_output_video_path}")
        display_frames = max(1, int((fps or self.fps) * float(display_sec)))
        shot_display_frames = max(1, int((fps or self.fps) * 1.5))
        minimap = MiniMapVisualizer() if draw_minimap_bounces else None
        self._text_patches = TextPatchRenderer()
        rally_starts = sorted(int(s) for s, _ in rally_spans) if rally_spans else []
        frame_index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_index += 1
            active_events = [
                event for event in events
                if 0 <= frame_index - int(event["frame"]) <= display_frames
            ]
            if draw_processed_trajectory:
                self.draw_processed_trajectory(frame, frame_index, trajectory_by_frame, trajectory_length=trajectory_length)
            for event in active_events:
                line_call = bounce_line_calls.get(int(event["frame"]))
                self.draw_event(frame, event, age_frames=frame_index - int(event["frame"]), display_frames=display_frames, line_call=line_call)
            if shot_hits:
                self.draw_active_shot_labels(frame, frame_index, shot_hits, shot_display_frames)
            if rally_starts and rally_label_pos is not None:
                self._draw_rally_count(frame, frame_index, rally_starts, rally_label_pos, rally_label_font_size)
            if minimap is not None and active_events:
                minimap.draw_bounce_events(frame, active_events)
            if minimap is not None and draw_processed_trajectory:
                current_point = trajectory_by_frame.get(frame_index)
                current_court = current_point.court if current_point is not None else None
                minimap.draw_processed_ball(frame, current_court)
            writer.write(frame)

        video.release()
        writer.release()
        encode_vscode_compatible_mp4(raw_output_video_path, output_video_path)
        if os.path.exists(raw_output_video_path):
            os.remove(raw_output_video_path)
        return True

    def draw_processed_trajectory(self, frame, frame_index, trajectory_by_frame, trajectory_length=30):
        points = []
        start_frame = max(1, frame_index - int(trajectory_length) + 1)
        for candidate_frame in range(start_frame, frame_index + 1):
            point = trajectory_by_frame.get(candidate_frame)
            if point is not None and point.image is not None:
                points.append(point)
        if not points:
            return

        for index, point in enumerate(points):
            x, y = int(point.image[0]), int(point.image[1])
            age_ratio = (index + 1) / len(points)
            color = (0, int(120 + 95 * age_ratio), 255)
            radius = max(2, int(2 + 5 * age_ratio))
            thickness = -1 if not point.interpolated else 1
            cv2.circle(frame, (x, y), radius, color, thickness, cv2.LINE_AA)

        latest = points[-1]
        x, y = int(latest.image[0]), int(latest.image[1])
        cv2.circle(frame, (x, y), 7, (0, 215, 255), -1, cv2.LINE_AA)

    # verdict -> BGR，out 红 / in 绿 / close 黄（复用弹跳标记本身的琥珀色，保持同一色系）
    LINE_CALL_COLORS = {"out": (0, 0, 255), "in": (0, 200, 0), "close": (0, 215, 255)}

    def draw_event(self, frame, event, age_frames=0, display_frames=1, line_call=None):
        image = event.get("image")
        if not image:
            return
        x, y = int(image[0]), int(image[1])
        progress = min(1.0, max(0.0, float(age_frames) / max(1.0, float(display_frames))))
        color = (0, 215, 255)
        radius = int(14 + 8 * progress)
        thickness = max(2, int(4 - 2 * progress))
        cv2.circle(frame, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Bounce {event.get('confidence', 0):.2f}",
            (x + 14, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        if line_call:
            verdict_color = self.LINE_CALL_COLORS.get(line_call, (255, 255, 255))
            cv2.putText(
                frame,
                line_call.upper(),
                (x + 14, max(24, y - 12) + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                verdict_color,
                2,
                cv2.LINE_AA,
            )

    # 击球标签的中文映射(面板已中文化,这里保持同口径)
    _HITTER_ZH = {"upper": "远端", "lower": "近处"}
    _SPIN_ZH = {"topspin": "上旋", "flat": "平击", "slice": "切削"}

    def draw_active_shot_labels(self, frame, frame_index, shot_hits, display_frames):
        """在击球帧起 display_frames 内画 "<hitter> <speed> km/h · <spin>"。

        中文经 TextPatchRenderer 预渲染贴片绘制——cv2.putText 只认 ASCII,直接画
        中文/·/— 会变成一串 "?"(踩过);速度/旋转拟合失败(fit_ok=False,含
        homography-only 降级)时整行只剩占位符,没有信息量,直接不画。
        """
        offset = 0
        for hit_frame, metric in sorted(shot_hits.items()):
            age = frame_index - int(hit_frame)
            if not (0 <= age <= display_frames):
                continue
            speed = metric.get("speed_kmh")
            spin_label = metric.get("spin_label")
            if speed is None and not spin_label:
                continue
            hitter = metric.get("hitter", "?")
            parts = [self._HITTER_ZH.get(hitter, hitter)]
            if speed is not None:
                parts.append(f"{speed:.0f} km/h")
            if spin_label:
                parts.append(self._SPIN_ZH.get(spin_label, spin_label))
            # "远端 87 km/h · 上旋":hitter 与数据段空格相连,数据段之间用 " · "
            text = parts[0] + " " + " · ".join(parts[1:])
            y = 22 + offset * 30
            patch = self._text_patches.render(text, 22, (255, 255, 255))
            if patch is not None:
                TextPatchRenderer.blit(frame, patch, (24, y))
            else:
                # 无中文字体的降级路径:纯 ASCII(远端/近处退回 far/near,分隔符用空格)
                ascii_parts = [{"upper": "far", "lower": "near"}.get(hitter, hitter)]
                if speed is not None:
                    ascii_parts.append(f"{speed:.0f} km/h")
                if spin_label:
                    ascii_parts.append(spin_label)
                cv2.putText(frame, " ".join(ascii_parts), (24, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            offset += 1

    def _draw_rally_count(self, frame, frame_index, rally_starts, label_pos, font_size):
        """画「回合: N」,N = start_frame <= 当前帧 的回合个数(视频开头第一拍之前不画)。"""
        count = bisect.bisect_right(rally_starts, frame_index)
        if count < 1:
            return
        size = int(font_size) if font_size else 24
        patch = self._text_patches.render(f"回合: {count}", size, (0, 165, 255))
        if patch is not None:
            TextPatchRenderer.blit(frame, patch, label_pos)
        else:
            cv2.putText(frame, f"Rally: {count}", (int(label_pos[0]), int(label_pos[1]) + size),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)

    def get_events(self):
        return list(self.events)

    def clear(self):
        self.events = []
        self.processed_points = []

    def _load_records(self, path):
        records = []
        skipped = 0
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, 1):
                line = line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
                    print(f"警告: 跳过 {path} 第 {line_number} 行损坏数据 / Warning: skipping corrupt line {line_number} in {path}")
        if skipped:
            print(f"警告: 共跳过 {skipped} 行损坏数据，弹跳分析基于剩余 {len(records)} 条记录 / Warning: skipped {skipped} corrupt lines; bounce analysis uses the remaining {len(records)} records")
        return records

    def _records_to_points(self, records):
        points = []
        for record in records:
            ball = record.get("tennis_ball") or {}
            points.append(
                TrajectoryPoint(
                    frame=int(record.get("frame", len(points) + 1)),
                    time_sec=float(record.get("time_sec") or 0.0),
                    image=self._point_or_none(ball.get("image")),
                    court=self._point_or_none(ball.get("court")),
                )
            )
        return points

    def _remove_outliers(self, points):
        # 空输入直接返回:np.array([]) 是 1-D,下面的 coords[:, 0] 会 IndexError
        # (生产踩过:整段视频没有一帧通过球场画面判定 → 零检测记录)
        if not points:
            return []
        cleaned = [TrajectoryPoint(**point.__dict__) for point in points]
        coords = np.array(
            [point.image if point.image is not None else [np.nan, np.nan] for point in cleaned],
            dtype=np.float32,
        )
        valid_indices = np.where(~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1]))[0]
        if len(valid_indices) < 5:
            return cleaned

        steps = []
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            frame_gap = max(1, cleaned[right].frame - cleaned[left].frame)
            steps.append(float(np.linalg.norm(coords[right] - coords[left]) / frame_gap))
        threshold = self._robust_threshold(np.array(steps, dtype=np.float32), floor=90.0)

        for index in valid_indices[1:-1]:
            prev_index = self._previous_valid(coords, index)
            next_index = self._next_valid(coords, index)
            if prev_index is None or next_index is None:
                continue
            prev_dist = float(np.linalg.norm(coords[index] - coords[prev_index]) / max(1, index - prev_index))
            next_dist = float(np.linalg.norm(coords[next_index] - coords[index]) / max(1, next_index - index))
            bridge_dist = float(np.linalg.norm(coords[next_index] - coords[prev_index]) / max(1, next_index - prev_index))
            isolated_jump = prev_dist > threshold and next_dist > threshold and bridge_dist < threshold
            if isolated_jump:
                cleaned[index].image = None
                cleaned[index].court = None
        return cleaned

    def _interpolate(self, points):
        interpolated = [TrajectoryPoint(**point.__dict__) for point in points]
        valid = [index for index, point in enumerate(interpolated) if point.image is not None]
        for left, right in zip(valid[:-1], valid[1:]):
            gap = right - left
            if gap <= 1 or gap - 1 > self.max_interpolation_gap:
                continue
            left_point = interpolated[left]
            right_point = interpolated[right]
            for index in range(left + 1, right):
                alpha = (index - left) / gap
                image = self._lerp(left_point.image, right_point.image, alpha)
                court = None
                if left_point.court is not None and right_point.court is not None:
                    court = self._lerp(left_point.court, right_point.court, alpha)
                interpolated[index].image = image
                interpolated[index].court = court
                interpolated[index].interpolated = True
        return interpolated

    def _velocity(self, coords):
        velocity = np.full(len(coords), np.nan, dtype=np.float32)
        for index in range(1, len(coords)):
            if np.isnan(coords[index]).any() or np.isnan(coords[index - 1]).any():
                continue
            velocity[index] = float(np.linalg.norm(coords[index] - coords[index - 1]) * self.fps)
        if len(velocity) > 1:
            velocity[0] = velocity[1]
        return velocity

    def _score_window(self, window, velocity, center, court_window=None):
        centered = window - np.nanmean(window, axis=0)
        scale = max(float(np.nanstd(centered)), 1.0)
        normalized = centered / scale
        smooth = self._smooth(window)
        center_point = smooth[center]
        before = smooth[max(0, center - 5):center]
        after = smooth[center + 1:min(len(smooth), center + 6)]
        if len(before) < 3 or len(after) < 3:
            return 0.0, {}

        before_center = np.mean(before, axis=0)
        after_center = np.mean(after, axis=0)
        v_in = center_point - before_center
        v_out = after_center - center_point
        turn_degrees = self._angle_between(v_in, v_out)
        deviation = self._point_line_distance(center_point, before_center, after_center)

        v_center = float(velocity[center])
        local_v = velocity[max(0, center - 4):min(len(velocity), center + 5)]
        median_v = float(np.nanmedian(local_v))
        peak_v = float(np.nanmax(local_v))
        speed_ratio = peak_v / max(median_v, 1.0)
        if v_center > self.max_center_velocity or speed_ratio > self.max_speed_ratio:
            return 0.0, {
                "reject_reason": "unstable_velocity",
                "center_velocity": round(float(v_center), 3),
                "speed_ratio": round(float(speed_ratio), 3),
                "window_size": int(self.window_size),
            }

        y = normalized[:, 1]
        y_slope_in = self._line_slope(np.arange(center + 1), y[:center + 1])
        y_slope_out = self._line_slope(np.arange(len(y) - center), y[center:])
        y_reversal = y_slope_in > 0.05 and y_slope_out < -0.05
        local_y_peak = window[center, 1] >= np.max(window[max(0, center - 5):min(len(window), center + 6), 1]) - 4.0
        local_y_valley = window[center, 1] <= np.min(window[max(0, center - 5):min(len(window), center + 6), 1]) + 4.0

        # 触地在图像里必是局部 y 最大(屏幕最低点)。局部 y 谷是过网点/弧顶,
        # 物理上不可能触地——落点图事故(job 6e98cc64 f16)就是过网点走谷通道混入。
        if not (y_reversal or local_y_peak):
            return 0.0, {
                "reject_reason": "no_ground_contact_signature",
                "turn_degrees": round(float(turn_degrees), 3),
                "deviation_px": round(float(deviation), 3),
                "local_y_valley": bool(local_y_valley),
                "window_size": int(self.window_size),
            }

        court_turn = 0.0
        court_deviation = 0.0
        if court_window is not None:
            court_smooth = self._smooth(court_window)
            court_center = court_smooth[center]
            court_before = np.mean(court_smooth[max(0, center - 5):center], axis=0)
            court_after = np.mean(court_smooth[center + 1:min(len(court_smooth), center + 6)], axis=0)
            court_turn = self._angle_between(court_center - court_before, court_after - court_center)
            court_deviation = self._point_line_distance(court_center, court_before, court_after)

            # 球拍击球否决:真弹跳后球继续朝同一深度方向走,只有击球才会让
            # court y 速度反向(与 segments 判 hit 同款信号)。两侧深度分量都
            # 超过 1 m/s 噪声地板才算有效反向——高吊近垂直下落时 vy≈0 不误伤。
            cy = court_smooth[:, 1]
            vy_in_ms = float(np.mean(np.diff(cy[:center + 1])[-5:])) * self.fps
            vy_out_ms = float(np.mean(np.diff(cy[center:])[:5])) * self.fps
            if vy_in_ms * vy_out_ms < 0 and min(abs(vy_in_ms), abs(vy_out_ms)) > 1.0:
                return 0.0, {
                    "reject_reason": "court_depth_reversal",
                    "vy_in_ms": round(vy_in_ms, 3),
                    "vy_out_ms": round(vy_out_ms, 3),
                    "turn_degrees": round(float(turn_degrees), 3),
                    "window_size": int(self.window_size),
                }

        angle_score = min(1.0, turn_degrees / 95.0)
        deviation_score = min(1.0, deviation / 18.0)
        speed_score = min(1.0, max(0.0, speed_ratio - 1.0) / 2.0)
        reversal_score = 1.0 if y_reversal else 0.0
        extreme_score = 1.0 if local_y_peak else 0.0
        court_score = max(min(1.0, court_turn / 75.0), min(1.0, court_deviation / 0.55))
        score = (
            0.28 * angle_score
            + 0.24 * deviation_score
            + 0.12 * speed_score
            + 0.16 * reversal_score
            + 0.10 * extreme_score
            + 0.10 * court_score
        )

        diagnostics = {
            "turn_degrees": round(float(turn_degrees), 3),
            "deviation_px": round(float(deviation), 3),
            "center_velocity": round(float(v_center), 3),
            "speed_ratio": round(float(speed_ratio), 3),
            "y_slope_in": round(float(y_slope_in), 4),
            "y_slope_out": round(float(y_slope_out), 4),
            "local_y_peak": bool(local_y_peak),
            "local_y_valley": bool(local_y_valley),
            "court_turn_degrees": round(float(court_turn), 3),
            "court_deviation_m": round(float(court_deviation), 3),
            "window_size": int(self.window_size),
        }
        return float(score), diagnostics

    def _valid_bounce_court_position(self, court):
        if court is None:
            return True
        try:
            x, y = float(court[0]), float(court[1])
        except (TypeError, IndexError, ValueError):
            return False
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        return (
            -self.court_margin <= x <= 10.97 + self.court_margin
            and -self.court_margin <= y <= 23.77 + self.court_margin
        )

    def _load_classifier(self):
        if self.classifier is not None or self.classifier_error is not None:
            return self.classifier
        if not self.classifier_path or not os.path.exists(self.classifier_path):
            self.classifier_error = f"classifier not found: {self.classifier_path}"
            return None
        if self.classifier_path.endswith(".cbm"):
            try:
                import catboost

                model = catboost.CatBoostRegressor()
                model.load_model(self.classifier_path)
                self.classifier = model
                self._classifier_kind = "catboost"
                print(f"已加载 CatBoost 弹跳模型 / Loaded CatBoost bounce model: {self.classifier_path}")
            except Exception as exc:
                self.classifier_error = f"{type(exc).__name__}: {exc}"
                print(f"CatBoost 弹跳模型加载失败，回退规则评分 / CatBoost bounce model load failed, falling back to rule-based scoring: {self.classifier_error}")
                return None
            return self.classifier
        try:
            self._install_pickle_compat()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(self.classifier_path, "rb") as file:
                    self.classifier = pickle.load(file)
            self._repair_legacy_classifier(self.classifier)
            self._classifier_kind = "sktime"
        except Exception as exc:
            self.classifier_error = f"{type(exc).__name__}: {exc}"
            print(f"寮硅烦鍒嗙被妯″瀷鍔犺浇澶辫触锛屽洖閫€瑙勫垯璇勫垎: {self.classifier_error}")
            return None
        print(f"宸插姞杞藉脊璺冲垎绫绘ā鍨? {self.classifier_path}")
        return self.classifier

    def _repair_legacy_classifier(self, classifier):
        estimators = []
        if hasattr(classifier, "steps"):
            estimators.extend(step for _, step in classifier.steps)
        estimators.append(classifier)
        for estimator in estimators:
            if not hasattr(estimator, "_is_vectorized"):
                estimator._is_vectorized = False
            if not hasattr(estimator, "_class_dictionary") and hasattr(estimator, "classes_"):
                estimator._class_dictionary = {
                    class_label: index for index, class_label in enumerate(list(estimator.classes_))
                }
            if not hasattr(estimator, "_y_metadata"):
                estimator._y_metadata = {"is_univariate": True}

    def _install_pickle_compat(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import sktime.transformations.panel.compose as panel_compose

            if not hasattr(panel_compose, "ColumnConcatenator"):
                panel_compose.ColumnConcatenator = LegacyColumnConcatenator
        except Exception:
            return

    def _window_to_feature_row(self, window, velocity):
        return {
            "x": pd.Series([float(window[index, 0]) for index in range(self.window_size)]),
            "y": pd.Series([float(window[index, 1]) for index in range(self.window_size)]),
            "V": pd.Series([float(velocity[index]) for index in range(self.window_size)]),
        }

    # 事件帧吸附半径:检测窗口滞后可能让中心帧偏离真实触地帧几帧,而空中球的
    # 单帧 homography 投影在深度方向误差可达数米(落点图事故根因之二)。
    SNAP_RADIUS_FRAMES = 4

    def _refine_events(self, points, events):
        """落点精化链(输入为未去重的原始候选,次序有讲究):
        1. 吸附:事件帧吸附到邻域内图像 y 最大(屏幕最低点=触地)的有效帧;
        2. 丢弃:最终帧处 court 深度方向速度反转 = 球拍击球,整个事件丢弃——
           评分层的否决只看以候选中心为准的窗口,中心偏前几帧的候选窗口里
           看不到反转,会漏过否决再被吸附拖到击球帧上(job 6e98cc64 f28/f142/f217);
        3. 去重:必须在丢弃**之后**——先去重会让高分击球点吃掉相邻真弹跳、
           自己再被丢,两头落空(job 6e98cc64 serve-1 落点);
        4. 精化:segments.refine_bounce 双抛物线求亚帧触地时刻、按该时刻插值
           court 坐标,替换单帧投影值。原始坐标保留在 diagnostics.court_raw。"""
        if not events:
            return events
        from .segments import refine_bounce

        dict_points = [
            {
                "frame": int(point.frame),
                "time_sec": float(point.time_sec),
                "image": list(point.image) if point.image is not None else None,
                "court": list(point.court) if point.court is not None else None,
            }
            for point in points
        ]
        by_frame = {p["frame"]: p for p in dict_points if p["image"] is not None}

        kept = []
        for event in events:
            frame = int(event["frame"])
            candidates = [
                f for f in range(frame - self.SNAP_RADIUS_FRAMES, frame + self.SNAP_RADIUS_FRAMES + 1)
                if f in by_frame
            ]
            if candidates:
                snapped = max(candidates, key=lambda f: by_frame[f]["image"][1])
                if snapped != frame:
                    point = by_frame[snapped]
                    event["frame"] = snapped
                    event["time_sec"] = round(point["time_sec"], 6)
                    event["image"] = [round(point["image"][0], 2), round(point["image"][1], 2)]
                    event["court"] = point["court"]
                    frame = snapped
            if self._depth_reversal_at(dict_points, frame):
                continue
            kept.append(event)

        refined_events = []
        for event in self._dedupe_events(kept):
            try:
                _, refined_court = refine_bounce(dict_points, int(event["frame"]))
            except (ValueError, TypeError):
                # 邻域内没有任何有效 court 坐标可插值时精化不可用,保留原值
                refined_court = None
            if refined_court is not None:
                diagnostics = event.setdefault("diagnostics", {})
                diagnostics["court_raw"] = event.get("court")
                event["court"] = [round(float(refined_court[0]), 2), round(float(refined_court[1]), 2)]
            refined_events.append(event)
        return refined_events

    def _depth_reversal_at(self, dict_points, frame, window=6):
        """最终帧两侧各 window 帧的 court y 端点斜率:真弹跳后球继续朝同一深度
        方向走,只有球拍击球才会反向。两侧都超过 1 m/s 噪声地板才算有效反向。"""
        court_by_frame = {
            p["frame"]: p["court"] for p in dict_points if p["court"] is not None
        }
        pre = [(f, court_by_frame[f][1]) for f in range(frame - window, frame + 1) if f in court_by_frame]
        post = [(f, court_by_frame[f][1]) for f in range(frame, frame + window + 1) if f in court_by_frame]
        if len(pre) < 3 or len(post) < 3:
            return False
        vy_in_ms = (pre[-1][1] - pre[0][1]) / max(1, pre[-1][0] - pre[0][0]) * self.fps
        vy_out_ms = (post[-1][1] - post[0][1]) / max(1, post[-1][0] - post[0][0]) * self.fps
        return vy_in_ms * vy_out_ms < 0 and min(abs(vy_in_ms), abs(vy_out_ms)) > 1.0

    def _dedupe_events(self, events):
        selected = []
        for event in sorted(events, key=lambda item: item["confidence"], reverse=True):
            if any(abs(event["frame"] - kept["frame"]) < self.min_event_gap_frames for kept in selected):
                continue
            selected.append(event)
        return sorted(selected, key=lambda item: item["frame"])

    def _write_events(self, path, events):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"events": events}, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def _write_trajectory(self, path, points):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "points": [
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)] if point.image is not None else None,
                    "court": [round(float(point.court[0]), 4), round(float(point.court[1]), 4)] if point.court is not None else None,
                    "interpolated": bool(point.interpolated),
                }
                for point in points
            ]
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def _rewrite_records_with_bounces(self, path, records, events):
        by_frame = {int(event["frame"]): event for event in events}
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            for record in records:
                record["bounce"] = by_frame.get(int(record.get("frame", -1)))
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                file.write("\n")
        os.replace(tmp_path, path)

    def _point_or_none(self, point):
        if point is None:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, IndexError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return [x, y]

    def _trajectory_by_frame(self, points):
        return {int(point.frame): point for point in points or [] if point.image is not None}

    def _previous_valid(self, coords, index):
        for candidate in range(index - 1, -1, -1):
            if not np.isnan(coords[candidate]).any():
                return candidate
        return None

    def _next_valid(self, coords, index):
        for candidate in range(index + 1, len(coords)):
            if not np.isnan(coords[candidate]).any():
                return candidate
        return None

    def _robust_threshold(self, values, floor):
        if len(values) == 0:
            return floor
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 6.0 * max(mad, 1.0))

    def _smooth(self, points):
        if len(points) < 3:
            return points
        smoothed = points.copy()
        for index in range(1, len(points) - 1):
            smoothed[index] = (points[index - 1] + points[index] * 2 + points[index + 1]) / 4.0
        return smoothed

    def _lerp(self, start, end, alpha):
        return [
            float(start[0] + (end[0] - start[0]) * alpha),
            float(start[1] + (end[1] - start[1]) * alpha),
        ]

    def _line_slope(self, x, y):
        if len(x) < 2:
            return 0.0
        return float(np.polyfit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), 1)[0])

    def _angle_between(self, vec_a, vec_b):
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom <= 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _point_line_distance(self, point, line_start, line_end):
        line = line_end - line_start
        denom = float(np.linalg.norm(line))
        if denom <= 1e-6:
            return float(np.linalg.norm(point - line_start))
        return float(abs(np.cross(line, point - line_start)) / denom)

