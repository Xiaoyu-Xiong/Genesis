from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="scene",
    target_file="src/scene.py",
    required_export="create_scene",
    responsibility=(
        "stage, fixed objects, global Genesis setup, FEM+IPC scene options, fixed props, fixed generated meshes, "
        "and scene lifecycle"
    ),
    prompt_body="""
    Write `create_scene(backend: str, *, sim_dt: float, sim_substeps: int, deformable_cfg: dict)`.
    The function must initialize Genesis and return an unbuilt `gs.Scene`.
    It must pass the supplied timing parameters into Genesis with
    `gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps)` when constructing the scene. Do not hardcode local
    timestep or substep defaults.
    If `deformable_cfg["enabled"]` is true and the Planner requests soft-body behavior, configure FEM+IPC through
    Genesis scene options. Use `deformable_cfg["genesis_precision"]` for `gs.init(...)` precision, and map IPC option
    values from `deformable_cfg` into `gs.options.IPCCouplerOptions(...)`. Do not use MPM, PBD, SPH, or rigid-only
    substitutes for soft-body tasks.
    If `deformable_cfg["enabled"]` is false, do not create FEM/IPC options. If the task fundamentally requires
    deformable physics, fail clearly in the worker report instead of writing a rigid approximation.
    Add at most one global ground Plane when the scene needs a floor. If you create it here, store the returned entity
    on `scene.genesis_static_floor` and describe it in `scene.genesis_case_metadata`; body.py must then reuse that
    scene-owned floor instead of adding a second coincident IPC plane. Never create duplicate overlapping ground planes
    in FEM+IPC scenes.
    Add a small number of fixed stage props suggested by the task, such as a wall, bin, ramp, stop, support,
    or ready fixed generated mesh from `assets/asset_manifest.json`. For fixed generated meshes, use the manifest
    runtime path, Genesis scale factors, and `file_meshes_are_zup` exactly; do not search the filesystem or infer
    orientation at runtime. Repaired generated mesh assets keep strict-manifold simulation geometry in `runtime_path`;
    `visual_path` is a seam-aware textured render mesh attached through
    `gs.morphs.Mesh(..., visual_file=entry["visual_path"], ...)`, not an independent simulation body. Keep fixed props
    lightweight: no more than 6 fixed objects.
    Do not create dynamic or task-moving bodies. Do not create cameras or render code.
    """,
)
