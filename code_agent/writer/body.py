from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="body",
    target_file="src/body.py",
    required_export="create_bodies",
    responsibility="movable rigid primitive or generated-mesh actors and task-participating bodies",
    prompt_body="""
    Write `create_bodies(scene, task: str)`.
    Return a list of dictionaries. Each dictionary must include:
    - `name`: string
    - `entity`: the Genesis entity returned by `scene.add_entity(...)`
    - `initial_velocity`: a 6-number tuple/list `(vx, vy, vz, wx, wy, wz)`
    Use dynamic rigid primitives and/or ready generated mesh assets from `assets/asset_manifest.json` when the Planner
    requested meshes. For each generated mesh, use the manifest runtime path, Genesis scale factors, and
    `file_meshes_are_zup` exactly; do not search the filesystem or infer orientation at runtime. Repaired generated
    mesh assets keep strict-manifold simulation geometry in `runtime_path`; `visual_path` is a seam-aware textured
    render mesh attached through `gs.morphs.Mesh(..., visual_file=entry["visual_path"], ...)`, not an independent
    simulation body.
    Do not split one generated object into separate simulation and visual mesh entities.
    Keep local GPU validation runs small: 3 to 8 dynamic bodies total.
    Include at least one projectile or mover with nonzero initial velocity for impact/scatter tasks.
    Do not call `scene.build()`, do not step the scene, and do not write artifacts.
    """,
)
