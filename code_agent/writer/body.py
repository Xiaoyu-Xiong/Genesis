from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="body",
    target_file="src/body.py",
    required_export="create_bodies",
    responsibility="movable rigid, FEM primitive/cloth, generated-mesh, XML/MJCF, and task-participating bodies",
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
    Use dynamic rigid primitives and/or ready generated mesh/cloth assets from `assets/asset_manifest.json` when the
    Planner requested assets. For each generated mesh or cloth mesh, use the manifest runtime path, Genesis scale
    factors, and
    `file_meshes_are_zup` exactly; do not search the filesystem or infer orientation at runtime. Repaired generated
    mesh assets keep strict-manifold simulation geometry in `runtime_path`; `visual_path` is a seam-aware textured
    render mesh attached through `gs.morphs.Mesh(..., visual_file=entry["visual_path"], ...)`, not an independent
    simulation body.
    For ready entries with `source_type == "cloth_mesh"`, create FEM shell cloth with
    `gs.morphs.Mesh(file=entry["runtime_path"], scale=entry["scale"] or 1.0, file_meshes_are_zup=True, ...)` and
    `gs.materials.FEM.Cloth(...)`. Use explicit cloth E, nu, rho, thickness, bending_stiffness, and friction_mu values
    from deformable_cfg cloth defaults/ranges. Do not tetrahedralize cloth meshes and do not pass tet_resolution for
    cloth morphs.
    If a generated mesh manifest entry is missing, has `status != "ready"`, reports failed validation, or fails Genesis
    import at runtime, fail clearly and ask Planner to regenerate that mesh asset through the mesh agent. Do not repair,
    simplify, retopologize, rescale, procedurally replace, or split the generated mesh inside body.py.
    For generated XML/MJCF articulated assets from `assets/asset_manifest.json`, load the canonical XML/MJCF path and
    expose a stable control contract for action.py: include actuator names, joint names, semantic DOF groups, control
    handles, and any required sign/axis notes that can be discovered from the manifest or source XML. If actuator or
    joint discovery must occur after `scene.build()`, expose enough semantic names and helper metadata for action.py to
    resolve them deterministically, and fail clearly if the requested mechanism cannot be controlled through the
    XML-designed actuators/DOFs.
    Do not split one generated object into separate simulation and visual mesh entities.
    If `deformable_cfg["enabled"]` is false and `deformable_cfg["ipc_enabled"]` is true, rigid and articulated bodies
    that should participate in IPC contact must use `gs.materials.Rigid(...)` coupling fields. Use
    `coup_type="ipc_only"` for simple non-articulated rigid objects fully handled by IPC, `coup_type="two_way_soft_constraint"`
    for Genesis-driven bodies or selected articulated contact links, and `coup_type="external_articulation"` for
    fixed-base articulated MJCF/URDF bodies that should couple at the DOF level. Keep all such objects rigid; do not
    fake soft deformation.
    If `deformable_cfg["ipc_enable_rigid_rigid_contact"]` is true for a heavy rigid-contact scene, treat the scene as
    pure IPC rigid-rigid contact: do not generate any `coup_type="two_way_soft_constraint"` body. Use `ipc_only` for
    passive free rigid bodies moved by IPC gravity/contact/friction/interlock, and use `external_articulation` for
    actively driven bodies that also contact IPC-owned rigid bodies. Do not rely on Genesis rigid contact to catch
    `ipc_only` objects; Genesis skips rigid-collider pairs involving `ipc_only` links. Passive IPC rigid bodies must not
    be directly pose-written, velocity-written, force-driven, hidden-welded, or attached after initialization.
    For an MJCF/URDF body that will be loaded as `coup_type="external_articulation"`, ensure it is a fixed-base
    articulation and every parent/child link that participates in a driven joint has collision geometry. If the fixed
    parent is only a logical mount, add a tiny nonzero-volume dummy collision geom to that parent. The dummy geom may be
    an MJCF primitive such as `type="box"`; it does not have to be a mesh. It must import as collision geometry and must
    not have both `contype` and `conaffinity` set to zero; `contype="1"` with `conaffinity="0"` is acceptable. Place the
    dummy geom far from real task contact so it cannot initially intersect active or passive bodies. Child/driven links
    must also have real collision geometry, with joint axes and pivots aligned to the physical layout.
    For FEM primitive soft-body tasks, create the requested soft primitive count when reasonable, but keep tet
    resolution from `deformable_cfg["tet_resolution"]` and avoid extra decorative dynamic bodies. A 10-soft-cube stack
    is acceptable for the primitive-first deformable suite.
    For deformable scenes, do not create a second ground/floor if scene.py already created one. Prefer reusing
    `getattr(scene, "genesis_static_floor", None)` in the returned actors list; only create a fallback floor if no
    scene-owned floor exists. Coincident IPC planes can crash UIPC collision filtering.
    For transparent rigid visual shells rendered with the native Genesis camera/rasterizer, do not rely on
    `gs.surfaces.Glass(color=(r, g, b, alpha))` for alpha transparency: Glass treats color as specular/transmission and
    the rasterizer path may drop the alpha channel. Use a rasterizer-blended surface such as
    `gs.surfaces.Smooth(color=(r, g, b), opacity=alpha, double_sided=True)` or another Plastic-derived surface with
    explicit `opacity`; keep the collision material/morph rigid and fixed as needed.
    All bodies participating in initial FEM+IPC contact must start without penetrations, self-intersections,
    coincident collision surfaces, or intentional overlap. This includes FEM primitives, generated FEM meshes, rigid
    collision props, fixed tubes/walls/floors, and articulated collision links. Use asset-manifest bbox/scale,
    primitive dimensions, and orientation to place bodies with positive clearance; for IPC scenes leave clearance at
    least `max(deformable_cfg.get("ipc_contact_d_hat", 0.01), 0.005)` unless the task explicitly begins at contact.
    For rotated boxes, use conservative rotated half-extents such as `0.5 * side * sqrt(3)` plus positive gap when
    computing stack heights; do not place a tilted bottom cube at exactly `side / 2` above the floor. If the scene
    needs compression or squeezing, let gravity/action/contact create it after the initial state is valid. When useful,
    include initial layout, bbox, and clearance metadata in returned actors so repair can inspect placement.
    Use `gs.materials.FEM.Elastic(...)` for volumetric soft primitives when `deformable_cfg["enabled"]` is true. Follow
    the common FEM material selection guide: pass explicit `E`, `nu`, and `rho`, keep them within the config ranges, and
    use config defaults when the task does not justify a special material. Choose explicit task-appropriate
    `friction_mu` values for FEM materials; do not read `deformable_cfg["fem_friction_mu"]`. Read FEM model,
    hydroelastic modulus, contact resistance, and hessian-invariant settings from deformable_cfg.
    Use morph `tet_resolution=deformable_cfg["tet_resolution"]` for FEM Box/Sphere/Cylinder primitives.
    Use `gs.materials.FEM.Cloth(...)` for thin sheet, ribbon, cylindrical shell, or spherical shell cloth tasks when
    `deformable_cfg["fem_cloth_enabled"]` is true and a ready `cloth_mesh` asset is available. PBD cloth remains out of
    scope: do not instantiate `gs.materials.PBD.Cloth`, `gs.options.PBDOptions`, or built-in `meshes/cloth.obj`.
    If deformable_cfg is disabled and the task fundamentally requires soft-body deformation, fail clearly instead of
    producing rigid substitutes.
    Include at least one projectile or mover with nonzero initial velocity for impact/scatter tasks.
    Do not call `scene.build()`, do not step the scene, and do not write artifacts.
    """,
)
