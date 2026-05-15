from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def render_initial_without_ipc(
    *,
    run_dir: Path,
    backend: str,
    out_dir: Path,
    deformable_config_path: Path | None = None,
    sim_dt: float = 0.01,
    sim_substeps: int = 1,
    render_fps: int = 25,
    render_res: tuple[int, int] = (960, 720),
) -> dict[str, Any]:
    """Render generated initial geometry with IPC scene coupling disabled.

    The body receives the original deformable/IPC configuration so body-side guards and material choices still reflect
    the generated scene. Only scene creation receives `ipc_enabled=False`, which avoids IPC coupler build-time
    intersection checks and gives the Planner/Critic a visual of the problematic initial layout.
    """

    started = time.time()
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "initial_no_ipc_render_report.json"

    report: dict[str, Any] = {
        "ok": False,
        "schema_version": 1,
        "diagnostic": "initial_no_ipc_render",
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "backend": backend,
        "created_at_unix": started,
        "scene_ipc_enabled": False,
        "body_receives_original_ipc_config": True,
    }

    try:
        main_module = _load_generated_main(run_dir)
        default_cfg = dict(getattr(main_module, "DEFAULT_DEFORMABLE_CFG", {}) or {})
        if deformable_config_path is not None and deformable_config_path.is_file():
            default_cfg.update(json.loads(deformable_config_path.read_text(encoding="utf-8")))

        adaptive = getattr(main_module, "_apply_adaptive_contact_d_hat", None)
        if callable(adaptive):
            adaptive(default_cfg, out_dir)

        body_cfg = dict(default_cfg)
        scene_cfg = dict(default_cfg)
        scene_cfg["ipc_enabled"] = False
        scene_cfg["ipc_contact_enable"] = False

        scene = main_module.create_scene(
            backend,
            sim_dt=float(sim_dt),
            sim_substeps=int(sim_substeps),
            deformable_cfg=scene_cfg,
        )
        actors = main_module.create_bodies(scene, getattr(main_module, "TASK", ""), deformable_cfg=body_cfg)
        render_state = main_module.setup_rendering(
            scene,
            actors,
            out_dir=out_dir,
            steps=0,
            fps=int(render_fps),
            duration_sec=None,
            target_video_frames=1,
            render_every_n_steps=1,
            render_res=render_res,
        )
        scene.build()
        capture = render_state.get("capture_frame")
        if callable(capture):
            capture(render_state, 0)
        elif hasattr(main_module, "capture_frame"):
            main_module.capture_frame(render_state, 0)
        else:
            raise RuntimeError("render_state did not expose capture_frame")
        render_stats = main_module.finalize_rendering(render_state)

        frames = [str(path) for path in sorted((out_dir / "frames").glob("frame_*.png"))]
        report.update(
            {
                "ok": bool(frames),
                "image_path": frames[0] if frames else None,
                "frames_dir": str(out_dir / "frames"),
                "video_path": str(out_dir / "render.mp4") if (out_dir / "render.mp4").is_file() else None,
                "render_stats_path": str(out_dir / "render_stats.json"),
                "render_stats": render_stats,
                "duration_sec": time.time() - started,
            }
        )
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_sec": time.time() - started,
            }
        )

    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _load_generated_main(run_dir: Path):
    src_dir = run_dir / "src"
    main_path = src_dir / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"Generated main.py not found: {main_path}")
    sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location("generated_initial_render_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated main.py from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render generated initial geometry with IPC scene coupling disabled.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--deformable-config", type=Path, default=None)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--sim-substeps", type=int, default=1)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--render-res", type=int, nargs=2, default=(960, 720))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = render_initial_without_ipc(
        run_dir=args.run_dir,
        backend=args.backend,
        out_dir=args.out_dir,
        deformable_config_path=args.deformable_config,
        sim_dt=args.sim_dt,
        sim_substeps=args.sim_substeps,
        render_fps=args.fps,
        render_res=(int(args.render_res[0]), int(args.render_res[1])),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
