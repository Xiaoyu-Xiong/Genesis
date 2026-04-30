from __future__ import annotations

import textwrap
from pathlib import Path

from code_agent.configs import CONFIGS


def write_main(
    *,
    run_dir: Path,
    task: str,
    default_steps: int,
    default_render_fps: int,
    default_duration_sec: float | None,
    default_target_video_frames: int | None,
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
    main_py.write_text(
        textwrap.dedent(
            f"""
            import argparse
            from pathlib import Path

            from action import run_actions
            from body import create_bodies
            from rendering import finalize_rendering, setup_rendering
            from scene import create_scene


            TASK = {task!r}


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
                parser.add_argument("--render", action="store_true", default=True)
                parser.add_argument("--no-render", action="store_false", dest="render")
                args = parser.parse_args()

                scene = create_scene(args.backend, sim_dt=args.sim_dt, sim_substeps=args.sim_substeps)
                actors = create_bodies(scene, TASK)
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
