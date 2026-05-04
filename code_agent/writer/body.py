from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="body",
    target_file="src/body.py",
    required_export="create_bodies",
    responsibility="movable rigid, FEM primitive, generated-mesh, XML/MJCF, and task-participating bodies",
    prompt_body="""
    Write `create_bodies(scene, task: str, *, deformable_cfg: dict)`.
    Return a list of dictionaries. Each dictionary must include:
    - `name`: string
    - `entity`: the Genesis entity returned by `scene.add_entity(...)`
    - `initial_velocity`: a 6-number tuple/list `(vx, vy, vz, wx, wy, wz)`
    For FEM/deformable actors, include:
    - `type`: a value such as `fem_soft_body`
    - `initial_velocity`: a 3-number tuple/list for FEM vertex velocity, or zeros
    - `material`: a short material description
    - `sample`: optional semantic sampling hints for action.py metrics
    Use dynamic rigid primitives and/or ready generated mesh assets from `assets/asset_manifest.json` when the Planner
    requested meshes. For each generated mesh, use the manifest runtime path, Genesis scale factors, and
    `file_meshes_are_zup` exactly; do not search the filesystem or infer orientation at runtime. Repaired generated
    mesh assets keep strict-manifold simulation geometry in `runtime_path`; `visual_path` is a seam-aware textured
    render mesh attached through `gs.morphs.Mesh(..., visual_file=entry["visual_path"], ...)`, not an independent
    simulation body.
    For generated XML/MJCF articulated assets from `assets/asset_manifest.json`, load the canonical XML/MJCF path and
    expose a stable control contract for action.py: include actuator names, joint names, semantic DOF groups, control
    handles, and any required sign/axis notes that can be discovered from the manifest or source XML. If actuator or
    joint discovery must occur after `scene.build()`, expose enough semantic names and helper metadata for action.py to
    resolve them deterministically, and fail clearly if the requested mechanism cannot be controlled through the
    XML-designed actuators/DOFs.
    Do not split one generated object into separate simulation and visual mesh entities.
    For FEM primitive soft-body tasks, create the requested soft primitive count when reasonable, but keep tet
    resolution from `deformable_cfg["tet_resolution"]` and avoid extra decorative dynamic bodies. A 10-soft-cube stack
    is acceptable for the primitive-first deformable suite.
    For deformable scenes, do not create a second ground/floor if scene.py already created one. Prefer reusing
    `getattr(scene, "genesis_static_floor", None)` in the returned actors list; only create a fallback floor if no
    scene-owned floor exists. Coincident IPC planes can crash UIPC collision filtering.
    FEM primitives must start without penetrations or self-intersections. For rotated boxes, use a conservative
    vertical half extent such as `0.5 * side * sqrt(3)` plus positive clearance/gap when computing stack heights; do
    not place a tilted bottom cube at exactly `side / 2` above the floor.
    Use `gs.materials.FEM.Elastic(...)` for soft primitives when `deformable_cfg["enabled"]` is true. Follow the common
    FEM material selection guide: pass explicit `E`, `nu`, and `rho`, keep them within the config ranges, and use config
    defaults when the task does not justify a special material. Read FEM model, hydroelastic modulus, friction, contact
    resistance, and hessian-invariant settings from deformable_cfg.
    Use morph `tet_resolution=deformable_cfg["tet_resolution"]` for FEM Box/Sphere/Cylinder primitives.
    If deformable_cfg is disabled and the task fundamentally requires soft-body deformation, fail clearly instead of
    producing rigid substitutes.
    Include at least one projectile or mover with nonzero initial velocity for impact/scatter tasks.
    Do not call `scene.build()`, do not step the scene, and do not write artifacts.
    """,
)
