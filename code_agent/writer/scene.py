from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="scene",
    target_file="src/scene.py",
    required_export="create_scene",
    responsibility="stage, fixed objects, global Genesis setup, fixed props, and artifact-neutral scene lifecycle",
    prompt_body="""
    Write `create_scene(backend: str)`.
    The function must initialize Genesis and return an unbuilt `gs.Scene`.
    Add a Plane and a small number of fixed stage props suggested by the task, such as a wall, bin, ramp, stop, or
    support. Keep fixed props lightweight: no more than 6 fixed primitives.
    Do not create dynamic or task-moving bodies. Do not create cameras or render code.
    """,
)
