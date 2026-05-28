from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code_agent.utils.local_execution import LocalRunConfig, run_local


def test_run_local_timeout_kills_child_process_tree(tmp_path: Path) -> None:
    main_path = tmp_path / "main.py"
    main_path.write_text(
        """
from __future__ import annotations

import subprocess
import sys
import time

child_code = (
    "from pathlib import Path; "
    "import time; "
    "time.sleep(1.5); "
    "Path('child_survived.txt').write_text('survived', encoding='utf-8')"
)
subprocess.Popen([sys.executable, "-c", child_code])
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )

    report = run_local(
        LocalRunConfig(
            workspace_dir=tmp_path,
            timeout_sec=0.3,
            python_executable="uv run --no-sync python",
        )
    )
    time.sleep(2.0)

    assert report["status"] == "timed_out"
    assert report["exit_code"] == 124
    assert report["timed_out"] is True
    assert report["timeout_process_tree"]["process_group"] is True
    assert not (tmp_path / "child_survived.txt").exists()


def test_run_local_filters_stale_artifacts(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    stale_metrics = artifacts_dir / "metrics.json"
    stale_video = artifacts_dir / "render.mp4"
    stale_metrics.write_text(json.dumps({"old": True}), encoding="utf-8")
    stale_video.write_bytes(b"old video")
    old_mtime = time.time() - 3600.0
    os.utime(stale_metrics, (old_mtime, old_mtime))
    os.utime(stale_video, (old_mtime, old_mtime))

    (tmp_path / "main.py").write_text(
        """
from __future__ import annotations

from pathlib import Path

Path("artifacts/summary.json").write_text('{"ok": true}\\n', encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )

    report = run_local(
        LocalRunConfig(
            workspace_dir=tmp_path,
            timeout_sec=5.0,
            python_executable=sys.executable,
            extra_artifact_paths=("artifacts",),
        )
    )

    artifact_paths = set(report["artifact_paths"])
    assert report["status"] == "passed"
    assert str(artifacts_dir / "summary.json") in artifact_paths
    assert str(stale_metrics) not in artifact_paths
    assert str(stale_video) not in artifact_paths
    assert report["artifacts"]["metrics"] is None
    assert report["artifacts"]["video"] is None
    assert report["stale_artifact_count"] == 2
    assert str(stale_metrics) in report["stale_artifact_paths_sample"]
    assert str(stale_video) in report["stale_artifact_paths_sample"]
