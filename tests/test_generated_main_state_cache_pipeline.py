from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_agent.utils.integrator import write_main


@pytest.mark.parametrize("mode", ["rigid", "articulated", "deformable", "cloth"])
def test_generated_main_state_cache_and_replay_pipeline_modes(tmp_path: Path, mode: str):
    run_dir = tmp_path / mode
    main_py = write_main(
        run_dir=run_dir,
        task=f"{mode} state cache smoke",
        default_steps=2,
        default_render_fps=3,
        default_duration_sec=1.0,
        default_target_video_frames=3,
        deformable_cfg={"enabled": mode in {"deformable", "cloth"}, "ipc_enabled": False},
    )
    _write_fake_case_modules(main_py.parent, mode)

    save_out = run_dir / "artifacts_save"
    _run_main(
        main_py,
        run_dir,
        "--out-dir",
        str(save_out),
        "--steps",
        "2",
        "--target-video-frames",
        "3",
        "--save-state-cache",
        "--require-state-cache",
        "--no-render",
    )
    manifest = save_out / "state_cache" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 3
    assert all((manifest.parent / frame["npz"]).is_file() for frame in payload["frames"])

    replay_out = run_dir / "artifacts_replay"
    _run_main(
        main_py,
        run_dir,
        "--out-dir",
        str(replay_out),
        "--steps",
        "2",
        "--target-video-frames",
        "3",
        "--replay-cache",
        str(manifest),
        "--render-only",
    )
    stats = json.loads((replay_out / "render_stats.json").read_text(encoding="utf-8"))
    assert stats["replay_only"] is True
    assert stats["physics_cache_manifest"] == str(manifest)
    assert stats["num_frames"] == 3
    assert len(list((replay_out / "frames").glob("frame_*.png"))) == 3
    if mode == "articulated":
        assert stats["captured_qpos"] == [[0.0, 0.0], [0.5, -0.25], [1.0, -0.5]]
        assert payload["actor_contracts"][0]["replay_mode"] == "qpos"
    if mode in {"deformable", "cloth"}:
        expected_z = [0.4, 0.5, 0.6] if mode == "deformable" else [0.1, 0.2, 0.3]
        assert stats["captured_state_z"] == pytest.approx(expected_z)
        assert payload["actor_contracts"][0]["replay_mode"] == "state_pos"


def _run_main(main_py: Path, run_dir: Path, *args: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(main_py), *args],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _write_fake_case_modules(src_dir: Path, mode: str) -> None:
    (src_dir / "scene.py").write_text(
        """
import genesis as gs


class FakeScene:
    def __init__(self, sim_dt, sim_substeps, rigid_options):
        self.dt = sim_dt
        self.t = 0.0
        self.built = False
        self.sim_options = gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps)
        self.rigid_options = rigid_options.model_copy_from(self.sim_options)

    def build(self):
        self.built = True

    def step(self):
        self.t += self.dt


def create_scene(backend: str, *, sim_dt: float, sim_substeps: int, rigid_options, deformable_cfg: dict):
    return FakeScene(sim_dt, sim_substeps, rigid_options)
""".lstrip(),
        encoding="utf-8",
    )
    (src_dir / "body.py").write_text(
        f"""
import numpy as np


class FakeState:
    def __init__(self, pos):
        self.pos = pos


class FakeEntity:
    def __init__(self, mode):
        self.mode = mode
        self.pos = np.array([1.0, 2.0, 3.0], dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.n_dofs = 2 if mode == "articulated" else 0
        self.n_links = 3 if mode == "articulated" else 1
        self.qpos = np.zeros(2, dtype=float) if mode == "articulated" else np.empty(0, dtype=float)
        z = 0.1 if mode == "cloth" else 0.4
        self.state_pos = (
            np.array([[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]], dtype=float)
            if mode in {{"deformable", "cloth"}}
            else np.empty((0, 3), dtype=float)
        )

    def get_pos(self):
        return self.pos

    def get_quat(self):
        return self.quat

    def set_pos(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def set_quat(self, quat):
        self.quat = np.asarray(quat, dtype=float)

    def get_qpos(self):
        return self.qpos.copy()

    def get_dofs_position(self):
        return self.qpos.copy()

    def set_qpos(self, qpos, *, zero_velocity=False):
        self.qpos = np.asarray(qpos, dtype=float)

    def set_dofs_position(self, position, *, zero_velocity=False):
        self.qpos = np.asarray(position, dtype=float)

    def set_position(self, pos):
        self.state_pos = np.asarray(pos, dtype=float)

    def get_state(self):
        return FakeState(self.state_pos.copy())


def create_bodies(scene, task: str, *, deformable_cfg: dict):
    return [{{"name": "{mode}_actor", "entity": FakeEntity("{mode}")}}]
""".lstrip(),
        encoding="utf-8",
    )
    (src_dir / "action.py").write_text(
        """
import json
from pathlib import Path


def run_actions(scene, actors, *, out_dir: Path, steps: int, render_state=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if render_state is not None:
        render_state["capture_frame"](render_state, 0)
    for step in range(1, steps + 1):
        scene.step()
        entity = actors[0]["entity"]
        if entity.mode == "articulated":
            entity.qpos += [0.5, -0.25]
        elif entity.mode in {"deformable", "cloth"}:
            entity.state_pos[:, 2] += 0.1
        if render_state is not None:
            render_state["capture_frame"](render_state, step)
    (out_dir / "event_log.json").write_text(json.dumps({"steps": steps, "samples": []}), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"success": True}), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"success": True}), encoding="utf-8")
    (out_dir / "run_result.json").write_text(json.dumps({"success": True}), encoding="utf-8")
    return {"success": True}
""".lstrip(),
        encoding="utf-8",
    )
    (src_dir / "rendering.py").write_text(
        """
import json
from pathlib import Path

from PIL import Image


def setup_rendering(
    scene,
    actors,
    *,
    out_dir: Path,
    steps: int,
    fps: int,
    duration_sec=None,
    target_video_frames=None,
    render_every_n_steps=1,
    render_res=(640, 480),
):
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture_steps = set(range(steps + 1)) if target_video_frames is None else set(range(min(steps + 1, target_video_frames)))
    state = {
        "out_dir": out_dir,
        "frames_dir": frames_dir,
        "capture_step_set": capture_steps,
        "capture_steps": capture_steps,
        "frame_paths": [],
        "fps": fps,
        "render_res": render_res,
        "duration_sec": duration_sec,
        "actors": actors,
        "captured_qpos": [],
        "captured_state_z": [],
    }
    state["capture_frame"] = capture_frame
    return state


def capture_frame(render_state: dict, step: int):
    if step not in render_state["capture_step_set"]:
        return
    frame_path = render_state["frames_dir"] / f"frame_{len(render_state['frame_paths']):03d}.png"
    Image.new("RGB", (8, 6), (20 + step, 30, 40)).save(frame_path)
    render_state["frame_paths"].append(frame_path)
    entity = render_state["actors"][0]["entity"]
    if entity.mode == "articulated":
        render_state["captured_qpos"].append(entity.get_qpos().tolist())
    elif entity.mode in {"deformable", "cloth"}:
        render_state["captured_state_z"].append(float(entity.get_state().pos[0, 2]))


def finalize_rendering(render_state: dict, *, event_log_path=None, metrics_path=None):
    out_dir = render_state["out_dir"]
    video_path = out_dir / "render.mp4"
    video_path.write_bytes(b"fake mp4")
    stats = {
        "rendered": True,
        "renderer": "fake",
        "video_path": str(video_path),
        "frames_dir": str(render_state["frames_dir"]),
        "num_frames": len(render_state["frame_paths"]),
        "fps": render_state["fps"],
        "duration_sec": render_state["duration_sec"],
        "render_res": list(render_state["render_res"]),
        "captured_qpos": render_state["captured_qpos"],
        "captured_state_z": render_state["captured_state_z"],
    }
    if "replay_only" in render_state:
        stats["replay_only"] = render_state["replay_only"]
        stats["physics_cache_manifest"] = render_state["physics_cache_manifest"]
    (out_dir / "render_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return stats
""".lstrip(),
        encoding="utf-8",
    )
