from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="body",
    target_file="src/body.py",
    required_export="create_bodies",
    responsibility="movable rigid primitive actors and task-participating bodies",
    prompt_body="""
    Write `create_bodies(scene, task: str)`.
    Return a list of dictionaries. Each dictionary must include:
    - `name`: string
    - `entity`: the Genesis entity returned by `scene.add_entity(...)`
    - `initial_velocity`: a 6-number tuple/list `(vx, vy, vz, wx, wy, wz)`
    Use only dynamic rigid primitive bodies. Keep local GPU validation runs small: 3 to 8 dynamic bodies.
    Include at least one projectile or mover with nonzero initial velocity for impact/scatter tasks.
    Do not call `scene.build()`, do not step the scene, and do not write artifacts.
    """,
)
