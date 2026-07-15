"""Baseline-local prompt clauses.

These are copied into the baseline so the end-to-end baseline prompt does not depend on the main code-agent prompt
modules. Keep edits local to this baseline when changing its behavior.
"""

PHYSICAL_CAUSALITY_CONTRACT = """
Physical causality contract:
- During simulation, every non-static object's motion must be explainable by the simulator's physical dynamics and by
  explicitly modeled control channels.
- After initialization, do not move task objects by direct pose/qpos/qvel writes, hidden constraints,
  target-following proxy forces, or outcome-directed external forces. A task object may move because of gravity,
  contact, friction, collision, joints, actuators, or an explicitly requested physical field/effect.
- If an object is manipulated by a robot or mechanism, the robot/mechanism should act through its actuators, joints,
  contacts, friction, or collision geometry. Do not servo the manipulated object itself toward an end-effector,
  handoff point, bin, platform, or goal pose.
- External forces are allowed only when they are part of the requested scene physics or a clearly modeled mechanism,
  such as wind, thrust, magnetism, springs, launch impulse, or actuator force. They must be applied to the physically
  appropriate body, recorded in metrics, and must not directly encode the desired task outcome.
- When a task requires an attachment-like effect, model it explicitly through geometry, joints, contacts, actuator
  force, or a task-requested physical mechanism. If that is not possible with the current assets/contracts, fail
  clearly instead of hiding the gap with object-following assistance.
""".strip()


COLLISION_CONTACT_CONTRACT = """
Collision/contact contract:
- Task-critical physical participants must have real collision geometry and enabled collision for expected contacts
  such as touching, rolling, sliding, stacking, blocking, catching, bouncing, or containment.
- Do not rely on visual-only geometry, pass-through objects, one-sided visual shells, hidden supports, or disabled
  collision masks to make the result look plausible.
- Avoid initial interpenetration. Use positive clearances, task-appropriate gaps, and metrics/logs that can reveal
  penetration, support loss, or ghost contact in contact-sensitive scenes.
- Robot internal self-collision may be limited when necessary to avoid actuator self-locking, but this must not disable
  collision between the robot/tool contact surfaces and the task objects they manipulate.
- For MJCF/URDF mechanisms coupled through IPC `coup_type="external_articulation"`, use fixed-base articulated DOFs and
  give every IPC-participating parent and child link real nonzero collision geometry. If a logical mount/parent would
  otherwise be empty, add a tiny primitive collision geom far outside the task contact region; never use only sites,
  inertial markers, visual-only geoms, zero-area planes, lines, or empty bodies for links that IPC must couple.
- Choose IPC rigid coupling mode by physical role: passive free rigid props use `ipc_only`, Genesis-driven contact links
  may use selected `two_way_soft_constraint`, and fixed-base driven mechanisms use `external_articulation`. Do not mix
  coupling modes casually in heavy rigid-rigid IPC contact scenes.
- When `ipc_enable_rigid_rigid_contact` / `enable_rigid_rigid_contact` is true for a heavy rigid-contact scene, treat
  the scene as pure IPC rigid-rigid contact: use `ipc_only` for passive rigid bodies and `external_articulation` for
  actively driven articulated bodies that contact IPC-owned rigid bodies. Avoid `two_way_soft_constraint` in heavy
  rigid-rigid IPC contact unless there is a specific non-heavy-contact reason.
- Use `coup_links` only with `two_way_soft_constraint`, and use it to couple just the contacting links such as gripper
  fingers instead of putting an entire robot into IPC.
- For `external_articulation`, dummy mount/base geometry is allowed only for IPC/articulation bookkeeping: it must be
  real nonzero collision geometry, named/reported, as small as practical, and far outside the main contact region with
  positive clearance from all task objects. It must not become a hidden obstacle, guide, stop, wedge, ramp, wall, or
  unilateral side obstacle. Active child/driven links such as rollers, clamps, sockets, paddles, and gripper fingers
  must remain collision-enabled with ordinary task-contact geometry.
- If `coup_collision_links` is used with fixed-child XML/MJCF mechanisms, account for fixed-joint merge behavior: include
  the merge target link in addition to semantic child contact links, otherwise visible contact parts may move while IPC
  collision geometry is filtered out.
- If logs mention IPC rigid ABD state retrieval/accessor errors, inspect stdout/stderr ordering first. When earlier
  libuipc diagnostics mention invalid initial geometry, `World is not valid`, intersection/distance checks, barrier or
  thickness failures, treat the ABD accessor message as secondary. Repair initial placement, spacing, duplicate IPC
  geometry, collision filters, missing geoms, scale/orientation, mesh/XML topology, or asset validity first.
- Preserve the intended IPC contact/coupling model while repairing geometry. Do not mask the issue by disabling IPC,
  disabling coupling, setting task-contact bodies out of the coupler, changing to hidden constraints, switching to
  rigid-only behavior, or bypassing contact unless a clean no-penetration repro still proves IPC capability is missing.
  Treat rigid ABD accessor support as a libuipc capability issue only after a valid rigid IPC scene with no
  world-validity, initial-penetration, distance, or thickness diagnostics still reproduces the accessor failure.
""".strip()


SCALE_POLICY_GUIDE = """
Scale policy:
- Avoid non-uniform scale by default. Unless the input task prompt explicitly requests an exceptional anisotropic
  scaling operation, do not use per-axis scale values such as `scale=(sx, sy, sz)` with unequal components for meshes,
  primitives, imported assets, MJCF/XML assets, or generated asset manifest requests.
- When an object needs a long, flat, thick, thin, or otherwise non-cubic shape, model that shape directly with
  primitive dimensions (`size`, `radius`, `height`), generated-asset geometry, XML primitive geom sizes, or a
  regenerated asset with the intended proportions. Do not create the intended proportions by stretching an already
  generated/imported mesh with non-uniform scale.
- Uniform scalar scale is acceptable for unit conversion, global sizing, and layout fitting. If a rare task genuinely
  requires non-uniform scale, document the reason and keep it isolated to that explicit requirement.
""".strip()


BUILTIN_ASSET_POLICY_GUIDE = """
Built-in Genesis asset policy:
- Do not inspect, copy, import, or reference prepackaged assets under `genesis/assets`.
- Do not use `gs.utils.get_assets_dir()`, `genesis.utils.misc.get_assets_dir()`, or Genesis built-in relative asset
  paths such as `xml/...`, `urdf/...`, or `meshes/...` for task geometry, robots, textures, or props.
- Use Genesis primitive morphs, case-workspace generated XML/MJCF assets, Meshy-generated assets, or explicit
  user-provided layout assets copied into the case workspace.
""".strip()


RENDER_CLARITY_GUIDE = """
Rendering clarity requirement:
- Generated videos must clearly show the task-relevant scene, objects, contacts, and motion. Camera position, lookat,
  field of view, clipping planes, lighting, background, capture cadence, and resolution should be chosen so the whole
  requested behavior is readable without severe cropping, tiny objects, occlusion, blank frames, or confusing views.
- If evidence says the result is hard to inspect because the camera is too far, too close, poorly aimed, poorly lit,
  cropped, static when tracking is needed, or otherwise unclear, actively repair the rendering and camera parameters.
- Record final camera parameters and rendering choices in render_stats.json so repairs can be source-aware.
""".strip()


GENERATED_RESULT_QUALITY_GUIDE = f"""
The final simulation should not merely satisfy numeric proxies. It should match the input text prompt and look
physically and visually reasonable, coherent, and logically staged.
{RENDER_CLARITY_GUIDE}
""".strip()


GENESIS_IMPLEMENTATION_GUIDE = """
Genesis implementation notes beyond the official docs:
- Consult the included Genesis documentation, examples, and local source before choosing APIs. Prefer documented names,
  argument names, renderer behavior, material/coupler options, and examples over guessed interfaces.
- Make initialization, timing, rendering, and scene construction follow the command-line arguments and generated
  contracts. Do not hard-code values that conflict with runner overrides or `contracts/deformable_config.json`.
- For generated mesh assets, read `assets/asset_manifest.json`. Use entries with `source_type == "generated_mesh"` and
  `status == "ready"`. Instantiate runtime geometry with `gs.morphs.Mesh(file=entry["runtime_path"],
  scale=entry["scale"] or 1.0, visual_file=entry.get("visual_path"),
  file_meshes_are_zup=entry.get("file_meshes_are_zup"), pos=(...), fixed=True/False)`.
- If a generated mesh manifest entry is missing, failed, invalid, or cannot be imported, fail clearly and regenerate
  the mesh asset; do not patch mesh files or silently replace it with primitives.
- FEM materials/entities are allowed only when `deformable_cfg["enabled"]` is true. If it is false, do not instantiate
  FEM materials/entities or deformation-only APIs.
- `gs.options.IPCCouplerOptions` is allowed when `deformable_cfg["ipc_enabled"]` is true. FEM deformable scenes force
  IPC on; rigid-only scenes may also use IPC when this flag is true.
- If `deformable_cfg["ipc_enabled"]` is false, do not instantiate `gs.options.IPCCouplerOptions`.
- Deformable scenes must stay in the FEM+IPC family. Do not use MPM, PBD, SPH, cloth-specific shortcuts, or rigid-only
  substitutes for soft-body prompts.
- Map IPC values from `deformable_cfg` by stripping the `ipc_` config prefix before passing Genesis option names.
  Exclude `ipc_contact_d_hat_adaptive`; it is a harness/runtime switch, not a Genesis option.
- Use exactly one IPC ground/support surface for the same contact region. Duplicate overlapping IPC planes are invalid
  initial geometry.
- FEM elastic materials should choose explicit `E`, `nu`, `rho`, and task-appropriate friction values from the
  deformable config ranges. Do not read a nonexistent `fem_friction_mu` override.
- FEM initialization may call `entity.set_velocity(...)` before stepping. After stepping starts, do not fake soft-body
  behavior with repeated `set_position` or `set_velocity` writes.
- FEM cloth is supported only through `gs.materials.FEM.Cloth(...)` plus `gs.morphs.Mesh(file=...)` surface meshes from
  ready cloth mesh asset-manifest entries. Do not instantiate `gs.materials.PBD.Cloth` for this baseline.
""".strip()
