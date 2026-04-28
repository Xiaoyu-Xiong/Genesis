from __future__ import annotations

import textwrap
from pathlib import Path

from code_agent.orchestration.generator import GeneratedProject


def write_main(*, run_dir: Path, task: str) -> GeneratedProject:
    """Write the stable entrypoint that wires Codex-authored modules together."""

    src_dir = run_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    main_py = src_dir / "main.py"
    main_py.write_text(
        textwrap.dedent(
            f"""
            import argparse
            from pathlib import Path

            from action import run_actions
            from body import create_bodies
            from rendering import render_outputs
            from scene import create_scene


            TASK = {task!r}


            def main():
                parser = argparse.ArgumentParser()
                parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
                parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
                parser.add_argument("--steps", type=int, default=40)
                parser.add_argument("--render", action="store_true", default=True)
                parser.add_argument("--no-render", action="store_false", dest="render")
                args = parser.parse_args()

                scene = create_scene(args.backend)
                actors = create_bodies(scene, TASK)
                scene.build()
                run_actions(scene, actors, out_dir=args.out_dir, steps=args.steps)
                if args.render:
                    render_outputs(
                        out_dir=args.out_dir,
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
    return GeneratedProject(src_dir=src_dir, main_py=main_py)
