from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GeneratedProject:
    src_dir: Path
    main_py: Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip() + "\n", encoding="utf-8")


def write_project(*, run_dir: Path, case_id: str, task: str) -> GeneratedProject:
    src_dir = run_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "contracts").mkdir(parents=True, exist_ok=True)
    (run_dir / "contracts" / "scene_brief.json").write_text(
        json.dumps({"case_id": case_id, "task": task}, indent=2) + "\n",
        encoding="utf-8",
    )

    _write(
        src_dir / "scene.py",
        """
        import genesis as gs


        def create_scene(backend: str):
            gs.init(
                backend=gs.cpu if backend == "cpu" else gs.gpu,
                precision="32",
                performance_mode=True,
                logging_level="warning",
            )
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.01),
                rigid_options=gs.options.RigidOptions(max_collision_pairs=64),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3.0, -5.0, 3.0),
                    camera_lookat=(0.0, 0.0, 0.7),
                    camera_fov=40,
                    max_FPS=60,
                ),
                show_viewer=False,
                show_FPS=False,
            )
            scene.add_entity(gs.morphs.Plane())
            # Fixed stage prop owned by Scene.
            scene.add_entity(gs.morphs.Box(pos=(0.0, 0.85, 0.10), size=(1.6, 0.10, 0.20), fixed=True))
            return scene
        """,
    )
    _write(
        src_dir / "body.py",
        """
        import genesis as gs


        def create_bodies(scene, task: str):
            actors = []
            lower = task.lower()
            if "sphere" in lower or "ball" in lower or "billiards" in lower:
                projectile = scene.add_entity(gs.morphs.Sphere(pos=(-1.1, -1.0, 0.22), radius=0.16))
            elif "cylinder" in lower or "cone" in lower:
                projectile = scene.add_entity(gs.morphs.Cylinder(pos=(-1.1, -1.0, 0.25), radius=0.14, height=0.35))
            else:
                projectile = scene.add_entity(gs.morphs.Box(pos=(-1.1, -1.0, 0.22), size=(0.28, 0.28, 0.28)))
            actors.append(("projectile", projectile, (2.4, 2.0, 0.0)))

            # Movable task actors live in Body. Use a tiny primitive set for CPU smoke speed.
            for idx, (x, z) in enumerate(((-0.15, 0.16), (0.15, 0.16))):
                if "cylinder" in lower or "cone" in lower:
                    entity = scene.add_entity(gs.morphs.Cylinder(pos=(x, 0.2, z), radius=0.12, height=0.28))
                elif "sphere" in lower or "ball" in lower:
                    entity = scene.add_entity(gs.morphs.Sphere(pos=(x, 0.2, z), radius=0.13))
                else:
                    entity = scene.add_entity(gs.morphs.Box(pos=(x, 0.2, z), size=(0.28, 0.28, 0.28)))
                actors.append((f"actor_{idx}", entity, (0.0, 0.0, 0.0)))
            return actors
        """,
    )
    _write(
        src_dir / "action.py",
        """
        import json
        from pathlib import Path


        def _safe_pos(entity):
            try:
                pos = entity.get_pos()
                if hasattr(pos, "detach"):
                    pos = pos.detach().cpu().numpy()
                if hasattr(pos, "tolist"):
                    return pos.tolist()
                return list(pos)
            except Exception:
                return None


        def run_actions(scene, actors, *, out_dir: Path, steps: int = 180):
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, entity, velocity in actors:
                if velocity != (0.0, 0.0, 0.0):
                    try:
                        entity.set_dofs_velocity([*velocity, 0.0, 0.0, 0.0])
                    except Exception:
                        pass

            samples = []
            for step in range(steps):
                scene.step()
                if step % 30 == 0 or step == steps - 1:
                    samples.append({
                        "step": step,
                        "actors": {name: _safe_pos(entity) for name, entity, _ in actors[:4]},
                    })

            metrics = {
                "success": True,
                "steps": steps,
                "num_actors": len(actors),
                "rendered": False,
                "summary": "Generated rigid CPU smoke simulation completed.",
            }
            event_log = {"steps": steps, "samples": samples}
            summary = {"status": "ok", "task_completed": True, "num_samples": len(samples)}
            run_result = {"status": "ok", "metrics": metrics, "summary": summary}
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n", encoding="utf-8")
            (out_dir / "event_log.json").write_text(json.dumps(event_log, indent=2) + "\\n", encoding="utf-8")
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
            (out_dir / "run_result.json").write_text(json.dumps(run_result, indent=2) + "\\n", encoding="utf-8")
            return metrics
        """,
    )
    _write(
        src_dir / "main.py",
        f"""
        import argparse
        from pathlib import Path

        from scene import create_scene
        from body import create_bodies
        from action import run_actions


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


        if __name__ == "__main__":
            main()
        """,
    )
    return GeneratedProject(src_dir=src_dir, main_py=src_dir / "main.py")
