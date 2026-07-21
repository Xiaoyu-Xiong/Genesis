from __future__ import annotations

import textwrap
from pathlib import Path

from code_agent.configs import CONFIGS, deformable_config_dict, rigid_config_dict
from code_agent.utils.codex import DEFAULT_REPO_ROOT


def write_main(
    *,
    run_dir: Path,
    task: str,
    default_steps: int,
    default_render_fps: int,
    default_sim_dt: float | None = None,
    default_sim_substeps: int | None = None,
    default_render_every_n_steps: int | None = None,
    default_render_res: tuple[int, int] | None = None,
    default_duration_sec: float | None,
    default_target_video_frames: int | None,
    rigid_cfg: dict[str, object] | None = None,
    deformable_cfg: dict[str, object] | None = None,
) -> Path:
    """Write the stable entrypoint that wires Codex-authored modules together."""

    src_dir = run_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    main_py = src_dir / "main.py"
    default_backend = CONFIGS.harness.default_backend
    resolved_sim_dt = CONFIGS.runtime.sim_dt if default_sim_dt is None else float(default_sim_dt)
    resolved_sim_substeps = CONFIGS.runtime.sim_substeps if default_sim_substeps is None else int(default_sim_substeps)
    resolved_render_every_n_steps = (
        CONFIGS.runtime.render_every_n_steps
        if default_render_every_n_steps is None
        else int(default_render_every_n_steps)
    )
    resolved_render_res = CONFIGS.runtime.render_res if default_render_res is None else tuple(default_render_res)
    default_rigid_cfg = dict(rigid_cfg or rigid_config_dict())
    default_deformable_cfg = dict(deformable_cfg or deformable_config_dict())
    main_py.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import inspect
            import json
            import sys
            from pathlib import Path

            REPO_ROOT = Path({str(DEFAULT_REPO_ROOT)!r})
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            from code_agent.utils.adaptive_ipc import adaptive_contact_d_hat_report, apply_adaptive_contact_d_hat
            from code_agent.utils.render_replay import run_render_only_replay
            from code_agent.utils.rigid_options import assert_scene_rigid_options, build_rigid_options
            from code_agent.utils.state_cache import (
                StateCacheWriter,
                attach_state_cache_capture,
                verify_state_cache_manifest,
            )

            from action import run_actions
            from body import create_bodies
            from rendering import finalize_rendering, setup_rendering
            from scene import create_scene


            TASK = {task!r}
            DEFAULT_RIGID_CFG = {default_rigid_cfg!r}
            DEFAULT_DEFORMABLE_CFG = {default_deformable_cfg!r}


            def _case_root() -> Path:
                return Path(__file__).resolve().parents[1]


            def _adaptive_contact_d_hat_report():
                return adaptive_contact_d_hat_report(
                    case_root=_case_root(),
                    default_deformable_cfg=DEFAULT_DEFORMABLE_CFG,
                    repo_root=REPO_ROOT,
                )


            def _apply_adaptive_contact_d_hat(deformable_cfg, out_dir: Path):
                return apply_adaptive_contact_d_hat(
                    deformable_cfg,
                    out_dir,
                    case_root=_case_root(),
                    default_deformable_cfg=DEFAULT_DEFORMABLE_CFG,
                    repo_root=REPO_ROOT,
                )


            def _call_with_optional_render_profile(func, *args, render_profile: str, **kwargs):
                try:
                    parameters = inspect.signature(func).parameters
                except (TypeError, ValueError):
                    parameters = {{}}
                if "render_profile" in parameters:
                    kwargs["render_profile"] = render_profile
                return func(*args, **kwargs)


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--backend", choices=("cpu", "gpu"), default={default_backend!r})
                parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
                parser.add_argument("--steps", type=int, default={int(default_steps)!r})
                parser.add_argument("--fps", "--render-fps", type=int, default={int(default_render_fps)!r})
                parser.add_argument("--duration-sec", type=float, default={default_duration_sec!r})
                parser.add_argument("--target-video-frames", type=int, default={default_target_video_frames!r})
                parser.add_argument("--sim-dt", type=float, default={float(resolved_sim_dt)!r})
                parser.add_argument("--sim-substeps", type=int, default={int(resolved_sim_substeps)!r})
                parser.add_argument("--render-every-n-steps", type=int, default={int(resolved_render_every_n_steps)!r})
                parser.add_argument("--render-res", type=int, nargs=2, default={list(resolved_render_res)!r})
                parser.add_argument("--deformable-config", type=Path, default=None)
                parser.add_argument(
                    "--render-profile",
                    choices=("debug_raster", "final_path_traced"),
                    default="debug_raster",
                )
                parser.add_argument("--save-state-cache", action="store_true", default=True)
                parser.add_argument("--require-state-cache", action="store_true", default=True)
                parser.add_argument("--replay-cache", type=Path, default=None)
                parser.add_argument("--render-only", action="store_true", default=False)
                parser.add_argument("--render", action="store_true", default=True)
                parser.add_argument("--no-render", action="store_false", dest="render")
                args = parser.parse_args()

                if args.render_only and args.replay_cache is None:
                    parser.error("--render-only requires --replay-cache")
                if args.render_only and not args.render:
                    parser.error("--render-only requires rendering to be enabled")

                replay_manifest = None
                if args.replay_cache is not None:
                    replay_manifest = verify_state_cache_manifest(
                        args.replay_cache,
                        require_npz=True,
                        require_complete_actor_state=True,
                    )
                run_steps = int(replay_manifest.get("steps", args.steps)) if isinstance(replay_manifest, dict) else args.steps
                replay_frame_count = (
                    len(replay_manifest.get("frames", [])) if isinstance(replay_manifest, dict) else None
                )
                target_video_frames = replay_frame_count if args.render_only else args.target_video_frames

                deformable_cfg = dict(DEFAULT_DEFORMABLE_CFG)
                if args.deformable_config is not None:
                    deformable_cfg.update(json.loads(args.deformable_config.read_text(encoding="utf-8")))
                _apply_adaptive_contact_d_hat(deformable_cfg, args.out_dir)
                rigid_options = build_rigid_options(DEFAULT_RIGID_CFG)

                scene = _call_with_optional_render_profile(
                    create_scene,
                    args.backend,
                    sim_dt=args.sim_dt,
                    sim_substeps=args.sim_substeps,
                    rigid_options=rigid_options,
                    deformable_cfg=deformable_cfg,
                    render_profile=args.render_profile,
                )
                assert_scene_rigid_options(scene, rigid_options)
                actors = _call_with_optional_render_profile(
                    create_bodies,
                    scene,
                    TASK,
                    deformable_cfg=deformable_cfg,
                    render_profile=args.render_profile,
                )
                render_state = None
                if args.render:
                    render_state = _call_with_optional_render_profile(
                        setup_rendering,
                        scene,
                        actors,
                        out_dir=args.out_dir,
                        steps=run_steps,
                        fps=args.fps,
                        duration_sec=args.duration_sec,
                        target_video_frames=target_video_frames,
                        render_every_n_steps=args.render_every_n_steps,
                        render_res=tuple(args.render_res),
                        render_profile=args.render_profile,
                    )

                if args.render_only:
                    replay_report = run_render_only_replay(
                        scene=scene,
                        actors=actors,
                        render_state=render_state,
                        cache_manifest=args.replay_cache,
                    )
                    if args.render:
                        stats = finalize_rendering(
                            render_state,
                            event_log_path=None,
                            metrics_path=None,
                        )
                        stats.update(replay_report)
                        stats["render_profile"] = args.render_profile
                        stats_path = args.out_dir / "render_stats.json"
                        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
                    return

                state_cache_writer = None
                if args.save_state_cache or args.require_state_cache:
                    state_cache_writer = StateCacheWriter.create(
                        out_dir=args.out_dir,
                        scene=scene,
                        actors=actors,
                        steps=args.steps,
                        render_state=render_state,
                        sim_dt=args.sim_dt,
                        sim_substeps=args.sim_substeps,
                        backend=args.backend,
                        render_profile=args.render_profile,
                    )
                    if render_state is None:
                        render_state = state_cache_writer.make_capture_state()
                    else:
                        render_state = attach_state_cache_capture(render_state, state_cache_writer)
                scene.build()
                run_actions(scene, actors, out_dir=args.out_dir, steps=args.steps, render_state=render_state)
                state_cache_manifest = None
                if state_cache_writer is not None:
                    state_cache_manifest = state_cache_writer.finalize()
                if args.require_state_cache:
                    verify_state_cache_manifest(
                        state_cache_manifest or args.out_dir / "state_cache" / "manifest.json",
                        require_npz=True,
                        require_complete_actor_state=True,
                    )
                if args.render:
                    stats = finalize_rendering(
                        render_state,
                        event_log_path=args.out_dir / "event_log.json",
                        metrics_path=args.out_dir / "metrics.json",
                    )
                    stats["render_profile"] = args.render_profile
                    if state_cache_manifest is not None:
                        stats["state_cache_manifest"] = str(state_cache_manifest)
                        stats["state_cache_required"] = bool(args.require_state_cache)
                    stats_path = args.out_dir / "render_stats.json"
                    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


            if __name__ == "__main__":
                main()
            """
        ).lstrip()
        + "\n",
        encoding="utf-8",
    )
    return main_py
