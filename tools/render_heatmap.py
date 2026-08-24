"""渲染原版球员位置热力图 PNG(worker 上传用)。

用法: uv run python tools/render_heatmap.py <output_dir>
读取 <output_dir>/detections.jsonl 与 metadata.json(取 fps),调用 zh 版
analyze_player_positions 生成 position_visualizations/heatmaps/match_heatmap.png,
成功时把该 PNG 路径打到 stdout 最后一行(worker 解析用),失败非零退出。

Render the original player-position heatmap PNG for worker upload. Forces the
Agg backend so it works headless under launchd.
"""
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")  # 必须在 matplotlib 导入前设置

# 与 tools/worker.py 同款:按路径调用时 sys.path[0] 只有 tools/,补仓库根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: render_heatmap.py <output_dir>", file=sys.stderr)
        return 2
    out_dir = sys.argv[1]
    detections = os.path.join(out_dir, "detections.jsonl")
    if not os.path.isfile(detections):
        print("detections.jsonl not found: %s" % detections, file=sys.stderr)
        return 1
    fps = 30.0
    meta_path = os.path.join(out_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                fps = float(json.load(f).get("video", {}).get("fps") or fps)
        except (ValueError, OSError):
            pass

    from tennis_analysis.visualization.player_positions_zh import analyze_player_positions

    viz_dir = os.path.join(out_dir, "position_visualizations")
    analyze_player_positions(detections, viz_dir, fps=fps)
    png = os.path.join(viz_dir, "heatmaps", "match_heatmap.png")
    if not os.path.isfile(png):
        print("match_heatmap.png not generated", file=sys.stderr)
        return 1
    print(png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
