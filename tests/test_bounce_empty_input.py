"""生产事故回归(job 60aa7c7f):is_court_view 拒掉全部帧 → detections.jsonl 为空 →
`_remove_outliers` 对 1-D 空数组做 `coords[:, 0]` 二维索引抛 IndexError,整单被标成
pipeline_error。空输入(零检测/零球点)必须优雅返回空结果,不允许崩溃。
"""
import json

from tennis_analysis.analysis.bounce import BounceDetector


def test_process_detections_on_empty_file_returns_no_events(tmp_path):
    detections_path = tmp_path / "detections.jsonl"
    detections_path.write_text("")
    detector = BounceDetector(fps=25.0)

    events = detector.process_detections(
        str(detections_path),
        output_path=str(tmp_path / "bounce_events.json"),
        trajectory_output_path=str(tmp_path / "cleaned_ball_trajectory.json"),
        rewrite_detections=True,
    )

    assert events == []
    assert detector.processed_points == []
    # 空结果也要按契约写出文件,下游 report_builder 读文件不读内存
    assert json.loads((tmp_path / "bounce_events.json").read_text()) == {"events": []}


def test_detect_from_points_empty_returns_no_events():
    assert BounceDetector(fps=25.0).detect_from_points([]) == []


def test_remove_outliers_empty_returns_empty():
    assert BounceDetector(fps=25.0)._remove_outliers([]) == []
