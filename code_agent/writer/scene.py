from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="scene",
    target_file="src/scene.py",
    required_export="create_scene",
    responsibility=(
        "stage, fixed objects, global Genesis setup, IPC/FEM scene options, fixed props, fixed generated/cloth meshes, "
        "and scene lifecycle"
    ),
    prompt_body="""
    Write `create_scene(backend: str, *, sim_dt: float, sim_substeps: int, rigid_options, deformable_cfg: dict,
    render_profile: str = "debug_raster")`.
    The function must initialize Genesis and return an unbuilt `gs.Scene`.
    It must pass the supplied timing parameters into Genesis with
    `gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps)` when constructing the scene. Do not hardcode local
    timestep or substep defaults.
    Pass `rigid_options=rigid_options` directly to `gs.Scene`. This object is constructed by the harness from
    `RigidConfigs`; do not instantiate `gs.options.RigidOptions`, mutate the supplied object, or override any of its
    fields in generated source.
    If `deformable_cfg["ipc_enabled"]` is true, configure IPC through Genesis scene options and map IPC option values
    from `deformable_cfg` into `gs.options.IPCCouplerOptions(...)`. This applies both to FEM+IPC deformable scenes and
    to rigid/articulated scenes whose contacts should be handled by IPC.
    Only pass actual `IPCCouplerOptions` fields to Genesis; do not pass config metadata keys ending in `_default`,
    `_min`, or `_max`.
    If `deformable_cfg["ipc_enabled"]` is false, do not create `gs.options.IPCCouplerOptions`.
    If `deformable_cfg["enabled"]` is true and the Planner requests soft-body or cloth behavior, keep the non-rigid
    parts in the FEM+IPC family. Use `deformable_cfg["genesis_precision"]` for `gs.init(...)` precision. Use
    `gs.materials.FEM.Cloth` for thin-shell cloth when requested and a ready cloth_mesh asset exists. Do not use MPM,
    PBD, SPH, or rigid-only substitutes for soft-body or cloth tasks.
    If `deformable_cfg["enabled"]` is false, do not create FEM materials/entities. If the task fundamentally requires
    deformable physics, fail clearly in the worker report instead of writing a rigid approximation.
    Add at most one global ground Plane when the scene needs a floor. If you create it here, store the returned entity
    on `scene.genesis_static_floor` and describe it in `scene.genesis_case_metadata`; body.py must then reuse that
    scene-owned floor instead of adding a second coincident IPC plane. Never create duplicate overlapping ground planes
    in FEM+IPC scenes.
    During ordinary physics debugging (`render_profile == "debug_raster"`), use the low-cost Rasterizer/native Genesis
    camera path with readable VisOptions. During the final render stage (`render_profile == "final_path_traced"` or
    `GENESIS_RENDER_PROFILE=final_path_traced`), construct the scene with GPU `gs.renderers.RayTracer`, a
    `WavePathIntegrator`/validated RayTracer integrator configuration when available, and metadata on the scene such as
    `scene.genesis_path_tracing` describing backend, integrator, spp, denoise, tracing depth, background/floor style,
    and lights, including each renderable light's position/radius or world bounds. Final path tracing should use
    area/sphere/mesh/emissive lights supported by RayTracer, not unsupported `scene.add_light` calls. Place all
    renderable light geometry outside the intended final camera frustum with enough clearance for any smoothed camera
    motion; a light background does not make a visible white light sphere acceptable.
    Keep the preferred final background light but deliberately separated in hue or value from prompt-required white
    subjects. Use coordinated non-white floors, walls, windows, or fixtures when needed to preserve silhouettes,
    contact shadows, folds, and motion readability.
    Add a small number of fixed stage props suggested by the task, such as a wall, bin, ramp, stop, support,
    or ready fixed generated mesh from `assets/asset_manifest.json`. For fixed generated meshes and cloth support
    meshes, use the manifest runtime path, Genesis scale factors, and `file_meshes_are_zup` exactly; do not search the
    filesystem or infer orientation at runtime. Repaired generated mesh assets keep strict-manifold simulation geometry
    in `runtime_path`;
    `visual_path` is a seam-aware textured render mesh attached through
    `gs.morphs.Mesh(..., visual_file=entry["visual_path"], ...)`, not an independent simulation body. Keep fixed props
    lightweight: no more than 6 fixed objects. If a generated mesh entry is missing, failed, or invalid, fail clearly
    and route regeneration to the mesh agent through Planner instead of editing or approximating the mesh in scene.py.
    Do not create dynamic or task-moving bodies. Do not create cameras or render code.
    """,
)
