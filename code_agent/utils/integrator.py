from __future__ import annotations

import textwrap
from pathlib import Path

from code_agent.configs import CONFIGS, deformable_config_dict


def write_main(
    *,
    run_dir: Path,
    task: str,
    default_steps: int,
    default_render_fps: int,
    default_duration_sec: float | None,
    default_target_video_frames: int | None,
    deformable_cfg: dict[str, object] | None = None,
) -> Path:
    """Write the stable entrypoint that wires Codex-authored modules together."""

    src_dir = run_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    main_py = src_dir / "main.py"
    default_backend = CONFIGS.harness.default_backend
    default_sim_dt = CONFIGS.runtime.sim_dt
    default_sim_substeps = CONFIGS.runtime.sim_substeps
    default_render_every_n_steps = CONFIGS.runtime.render_every_n_steps
    default_render_res = CONFIGS.runtime.render_res
    default_deformable_cfg = dict(deformable_cfg or deformable_config_dict())
    main_py.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            from pathlib import Path

            import numpy as np
            import trimesh

            from action import run_actions
            from body import create_bodies
            from rendering import finalize_rendering, setup_rendering
            from scene import create_scene


            TASK = {task!r}
            DEFAULT_DEFORMABLE_CFG = {default_deformable_cfg!r}


            def _case_root() -> Path:
                return Path(__file__).resolve().parents[1]


            def _resolve_case_path(path_value):
                path = Path(path_value)
                if path.is_absolute():
                    return path
                for candidate in (Path.cwd() / path, _case_root() / path):
                    if candidate.exists():
                        return candidate
                return _case_root() / path


            def _mesh_edge_lengths(asset):
                runtime_path = asset.get("runtime_path")
                if not runtime_path:
                    return np.empty(0, dtype=float)
                mesh_path = _resolve_case_path(runtime_path)
                if not mesh_path.is_file():
                    return np.empty(0, dtype=float)
                mesh = trimesh.load_mesh(str(mesh_path), force="mesh", process=False, skip_texture=True)
                if hasattr(mesh, "dump"):
                    dumped = mesh.dump(concatenate=True)
                    if dumped is not None:
                        mesh = dumped
                vertices = np.asarray(mesh.vertices, dtype=float)
                edges = np.asarray(mesh.edges_unique, dtype=np.int64)
                if vertices.size == 0 or edges.size == 0:
                    return np.empty(0, dtype=float)
                scale = asset.get("scale")
                if isinstance(scale, (list, tuple)) and len(scale) == 3:
                    vertices = vertices * np.asarray(scale, dtype=float)
                lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
                return lengths[np.isfinite(lengths) & (lengths > 0.0)]


            def _adaptive_contact_d_hat_report():
                manifest_path = _resolve_case_path("assets/asset_manifest.json")
                if not manifest_path.is_file():
                    return None
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                assets = manifest.get("assets")
                if not isinstance(assets, list):
                    return None
                length_chunks = []
                source_assets = []
                for asset in assets:
                    if not isinstance(asset, dict):
                        continue
                    if asset.get("source_type") != "generated_mesh" or asset.get("status") != "ready":
                        continue
                    lengths = _mesh_edge_lengths(asset)
                    if lengths.size == 0:
                        continue
                    length_chunks.append(lengths)
                    source_assets.append(str(asset.get("logical_name") or asset.get("runtime_path")))
                if not length_chunks:
                    return None
                lengths = np.concatenate(length_chunks)
                median_edge_length = float(np.median(lengths))
                return {{
                    "source": "assets/asset_manifest.json",
                    "rule": "ipc_contact_d_hat = 0.75 * median generated-mesh edge length",
                    "source_assets": source_assets,
                    "edge_count": int(lengths.size),
                    "median_edge_length": median_edge_length,
                    "ipc_contact_d_hat": 0.75 * median_edge_length,
                }}


            def _apply_adaptive_contact_d_hat(deformable_cfg, out_dir: Path):
                if not bool(deformable_cfg.get("ipc_contact_d_hat_adaptive", False)):
                    return None
                report = _adaptive_contact_d_hat_report()
                if report is None:
                    print("[adaptive-ipc] no ready generated mesh edges found; keeping configured ipc_contact_d_hat")
                    return None
                deformable_cfg["ipc_contact_d_hat"] = float(report["ipc_contact_d_hat"])
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "adaptive_ipc_config.json").write_text(
                    json.dumps(report, indent=2),
                    encoding="utf-8",
                )
                print(
                    "[adaptive-ipc] "
                    f"ipc_contact_d_hat={{report['ipc_contact_d_hat']:.6g}} "
                    f"from median_edge_length={{report['median_edge_length']:.6g}} "
                    f"over {{report['edge_count']}} edges"
                )
                return report


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--backend", choices=("cpu", "gpu"), default={default_backend!r})
                parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
                parser.add_argument("--steps", type=int, default={int(default_steps)!r})
                parser.add_argument("--fps", "--render-fps", type=int, default={int(default_render_fps)!r})
                parser.add_argument("--duration-sec", type=float, default={default_duration_sec!r})
                parser.add_argument("--target-video-frames", type=int, default={default_target_video_frames!r})
                parser.add_argument("--sim-dt", type=float, default={float(default_sim_dt)!r})
                parser.add_argument("--sim-substeps", type=int, default={int(default_sim_substeps)!r})
                parser.add_argument("--render-every-n-steps", type=int, default={int(default_render_every_n_steps)!r})
                parser.add_argument("--render-res", type=int, nargs=2, default={list(default_render_res)!r})
                parser.add_argument("--deformable-config", type=Path, default=None)
                parser.add_argument("--render", action="store_true", default=True)
                parser.add_argument("--no-render", action="store_false", dest="render")
                args = parser.parse_args()

                deformable_cfg = dict(DEFAULT_DEFORMABLE_CFG)
                if args.deformable_config is not None:
                    deformable_cfg.update(json.loads(args.deformable_config.read_text(encoding="utf-8")))
                _apply_adaptive_contact_d_hat(deformable_cfg, args.out_dir)

                scene = create_scene(
                    args.backend,
                    sim_dt=args.sim_dt,
                    sim_substeps=args.sim_substeps,
                    deformable_cfg=deformable_cfg,
                )
                actors = create_bodies(scene, TASK, deformable_cfg=deformable_cfg)
                render_state = None
                if args.render:
                    render_state = setup_rendering(
                        scene,
                        actors,
                        out_dir=args.out_dir,
                        steps=args.steps,
                        fps=args.fps,
                        duration_sec=args.duration_sec,
                        target_video_frames=args.target_video_frames,
                        render_every_n_steps=args.render_every_n_steps,
                        render_res=tuple(args.render_res),
                    )
                scene.build()
                run_actions(scene, actors, out_dir=args.out_dir, steps=args.steps, render_state=render_state)
                if args.render:
                    finalize_rendering(
                        render_state,
                        event_log_path=args.out_dir / "event_log.json",
                        metrics_path=args.out_dir / "metrics.json",
                    )


            if __name__ == "__main__":
                main()
            """
        ).lstrip()
        + "\n",
        encoding="utf-8",
    )
    return main_py
