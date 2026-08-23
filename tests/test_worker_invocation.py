"""README/docstring 都写 `uv run tools/worker.py --once`——直接以脚本路径调用。

回归点:worker.py 用 `from tools.report_builder import ...` 绝对导入,但脚本按路径调用时
Python 只把 tools/ 自身目录塞进 sys.path[0],仓库根目录不在 sys.path,导致
`ModuleNotFoundError: No module named 'tools'`,文档写的调用方式直接炸。
用 --help 触发到顶层 import 又不需要 RV_WORKER_TOKEN / 网络,是最小复现路径。
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_worker_script_runs_when_invoked_by_path():
    proc = subprocess.run(
        [sys.executable, "tools/worker.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
