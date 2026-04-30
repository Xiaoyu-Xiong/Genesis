from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="scene",
    target_file="src/scene.py",
    required_export="create_scene",
    responsibility="stage, fixed objects, global Genesis setup, fixed props, fixed generated meshes, and scene lifecycle",
    prompt_body="""
    Write `create_scene(backend: str)`.
    The function must initialize Genesis and return an unbuilt `gs.Scene`.
    Add a Plane and a small number of fixed stage props suggested by the task, such as a wall, bin, ramp, stop, support,
    or ready fixed generated mesh from `assets/asset_manifest.json`. For fixed generated meshes, use the manifest
    runtime path, Genesis scale factors, and `file_meshes_are_zup` exactly; do not search the filesystem or infer
    orientation at runtime. Repaired generated mesh assets keep strict-manifold simulation geometry in `runtime_path`;
    `visual_path` is a seam-aware textured render mesh attached through
    `gs.morphs.Mesh(..., visual_file=entry["visual_path"], ...)`, not an independent simulation body. Keep fixed props
    lightweight: no more than 6 fixed objects.
    Do not create dynamic or task-moving bodies. Do not create cameras or render code.
    """,
)
