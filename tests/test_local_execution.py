from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required for render video normalization tests",
)
def test_run_local_reencodes_render_video_from_saved_frames(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    (tmp_path / "main.py").write_text(
        f"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

artifacts = Path("artifacts")
frames_dir = artifacts / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
    Image.new("RGB", (32, 24), color).save(frames_dir / f"frame_{{index:03d}}.png")

subprocess.run(
    [
        {str(ffmpeg)!r},
        "-y",
        "-v",
        "error",
        "-loop",
        "1",
        "-i",
        str(frames_dir / "frame_000.png"),
        "-frames:v",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(artifacts / "render.mp4"),
    ],
    check=True,
)
(artifacts / "render_stats.json").write_text(
    json.dumps(
        {{
            "rendered": True,
            "video_path": "artifacts/render.mp4",
            "frames_dir": "artifacts/frames",
            "num_frames": 3,
            "fps": 25,
            "duration_sec": 3 / 25,
            "video_writer_strategy": "genesis_camera_recording",
        }},
        indent=2,
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
""".lstrip(),
        encoding="utf-8",
    )

    report = run_local(
        LocalRunConfig(
            workspace_dir=tmp_path,
            timeout_sec=10.0,
            python_executable=sys.executable,
            extra_artifact_paths=("artifacts",),
        )
    )

    assert report["status"] == "passed"
    normalization = report["render_video_normalization"]
    assert normalization["changed"] is True
    assert normalization["strategy"] == "harness_ffmpeg_from_png_frames"
    assert normalization["before_probe"]["frame_count"] == 1
    assert normalization["after_probe"]["frame_count"] == 3
    assert _probe_frame_count(tmp_path / "artifacts" / "render.mp4") == 3

    stats = json.loads((tmp_path / "artifacts" / "render_stats.json").read_text(encoding="utf-8"))
    assert stats["video_writer_strategy"] == "harness_ffmpeg_from_png_frames"
    assert stats["video_reencoded_from_frames"] is True
    assert stats["video_frame_count_verified"] == 3


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe are required for render video validation tests",
)
def test_run_local_rejects_short_render_video_without_saved_frames(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    (tmp_path / "main.py").write_text(
        f"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

artifacts = Path("artifacts")
artifacts.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (32, 24), (255, 0, 0)).save(artifacts / "single.png")
subprocess.run(
    [
        {str(ffmpeg)!r},
        "-y",
        "-v",
        "error",
        "-loop",
        "1",
        "-i",
        str(artifacts / "single.png"),
        "-frames:v",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(artifacts / "render.mp4"),
    ],
    check=True,
)
(artifacts / "render_stats.json").write_text(
    json.dumps(
        {{
            "rendered": True,
            "video_path": "artifacts/render.mp4",
            "frames_dir": "artifacts/frames",
            "num_frames": 3,
            "fps": 25,
            "video_writer_strategy": "genesis_camera_recording",
        }},
        indent=2,
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
""".lstrip(),
        encoding="utf-8",
    )

    report = run_local(
        LocalRunConfig(
            workspace_dir=tmp_path,
            timeout_sec=10.0,
            python_executable=sys.executable,
            extra_artifact_paths=("artifacts",),
        )
    )

    assert report["status"] == "failed"
    assert report["exit_code"] == 1
    assert report["process_exit_code"] == 0
    assert report["artifact_validation_failed"] is True
    assert "video artifact has 1 frames" in report["artifact_validation"]["errors"][0]


def _probe_frame_count(path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return int(completed.stdout.strip())
