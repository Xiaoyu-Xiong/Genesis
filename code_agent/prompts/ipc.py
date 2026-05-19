"""FEM, IPC, and rigid-coupling prompt guides."""

FEM_MATERIAL_SELECTION_GUIDE = """
FEM material selection guide:
- Generated FEM elastic materials must pass explicit `E`, `nu`, and `rho` to `gs.materials.FEM.Elastic(...)`.
- Choose material values for the task within the ranges exposed in `deformable_cfg`: `fem_youngs_modulus_min/max`,
  `fem_poisson_ratio_min/max`, and `fem_density_min/max`. If the task does not justify a special material, use
  `fem_youngs_modulus_default`, `fem_poisson_ratio_default`, and `fem_density_default`.
- `E` is Young's modulus in Pascals. `1e4` to `5e4` is very soft jelly or gel with large visible deformation; `5e4` to
  `5e5` is soft rubber with clear wobble and compression; `5e5` to `5e6` is firmer elastomer or soft plastic with smaller
  deformation.
- `nu` is Poisson ratio. Around `0.2` is more compressible or foam-like; `0.35` is a balanced soft-solid default; `0.45`
  is nearly incompressible, volume-preserving rubber and can be numerically harder.
- `rho` is density in kg/m^3. Around `300` is light foam-like material; `1000` is a water-like gel/rubber default; `3000`
  is a heavy dense soft solid.
- Choose explicit `friction_mu` values for each FEM material. Do not read FEM friction from `deformable_cfg`. Around
  `0.0` to `0.1` is slippery or nearly frictionless contact, `0.2` to `0.5` is typical soft rubber/plastic contact, and
  `0.6` to `1.0` is high-friction sticky contact; use values outside that range only when the task clearly justifies it.
""".strip()


RIGID_IPC_COUPLING_GUIDE = """
Rigid IPC coupling mode guide:
- `coup_type="ipc_only"` means the rigid non-articulated object is simulated by IPC for gravity and contact, then its
  transform is copied back to Genesis for rendering/state queries. Use it for passive rigid props, loose rigid links,
  simple obstacles, chain links, anchors, balls, or boxes whose motion should mainly come from IPC contact. It is not
  supported for articulated objects, and many direct post-build pose/velocity control APIs are unavailable or
  inappropriate because IPC owns the motion.
- `coup_type="two_way_soft_constraint"` keeps a Genesis rigid/articulated body driven by Genesis dynamics or controls
  while IPC tracks it with a soft transform constraint and can feed contact forces/torques back when
  `IPCCouplerOptions.two_way_coupling` is true. Use it for actuator-driven rigid bodies, moving tools, gripper fingers,
  robot links, windup drums, paddles, presses, or selected articulated links that need IPC contact but still need
  Genesis controls. `constraint_strength_translation` and `constraint_strength_rotation` tune how tightly IPC follows
  Genesis: higher is stiffer and less laggy, lower is softer and usually more forgiving.
- `coup_type="external_articulation"` couples a fixed-base articulated MJCF/URDF entity at the joint/DOF level through
  IPC. Use it when the whole articulated mechanism should participate in IPC contact according to its joint state, such
  as a robot arm or gripper represented as one MJCF asset. It is stricter than `two_way_soft_constraint`: avoid
  post-build root/qpos teleports, drive motion through actuator/DOF controls, and be careful with initialization because
  some direct state-setting APIs are unsupported.
- When `ipc_enable_rigid_rigid_contact` / `enable_rigid_rigid_contact` is true for a heavy rigid-contact scene, treat
  the scene as pure IPC rigid-rigid contact: do not create any rigid body with
  `coup_type="two_way_soft_constraint"`. Use `coup_type="ipc_only"` for free passive rigid bodies whose motion comes
  from gravity/contact/friction/interlock, and use `coup_type="external_articulation"` for actively driven bodies that
  must also contact IPC-owned rigid bodies.
- Do not rely on Genesis rigid contact as a fallback for `ipc_only` objects. Genesis's rigid collider skips pairs
  involving `ipc_only` links; rigid-rigid contact between such bodies must be handled by IPC. Avoid mixing
  `two_way_soft_constraint` and `ipc_only` in heavy interlocking contact because the soft transform constraint and
  Genesis/IPC state synchronization can create inconsistent states, sudden ejection, or crashes.
- Passive IPC rigid bodies must not be pose-written, velocity-written, force-driven, hidden-welded, or directly
  attached after initialization. Drive only the intended articulation DOFs of active mechanisms, not the passive rigid
  payloads they contact.
- `coup_links=(...)` is only for `two_way_soft_constraint`; use it to couple just the links that contact the task
  object, such as left/right gripper fingers, instead of putting an entire robot into IPC.
- If `coup_type` is left as `None`, Genesis auto-selects a mode based on entity type, but generated code should choose
  an explicit mode when the task's contact behavior depends on it.
""".strip()


EXTERNAL_ARTICULATION_MJCF_GUIDE = """
External-articulation MJCF/XML guide for IPC:
- An MJCF/URDF entity used with `coup_type="external_articulation"` must be a fixed-base articulation, with a fixed
  parent/body and a revolute or prismatic child/body for each driven mechanism.
- Every link that participates in an external-articulation joint must have collision geometry. An empty logical parent
  body can make IPC fail to create an ABD slot for that link.
- If the fixed parent is only a logical mount, add a tiny dummy collision geometry to that parent. The dummy parent geom
  does not have to be a mesh; a primitive MJCF geom such as `type="box"` is fine.
- The dummy geom must be a real nonzero-volume collision geom, not a site, inertial, visual-only marker, zero-area
  plane, line, or empty body. It must participate in collision semantics: do not set both `contype` and `conaffinity`
  to zero. A useful pattern is `contype="1"` with `conaffinity="0"` so the geom remains a collision geom without
  accepting ordinary contact pairs.
- Place dummy mount geometry far from the real contact region so it cannot initially intersect the active mechanism,
  passive rigid bodies, ground, chain links, anchors, or other task geometry. It may be invisible/transparent, but it
  must still import as collision geometry.
- Child/driven geoms, such as a spool or hinge tool, must have real collision geometry; do not provide only visual
  geometry for IPC contact participants.
- Keep XML mesh coordinates aligned with the layout/source mesh coordinates. If a fixed `euler` correction is required
  on a geom, validate the resulting initial configuration with an IPC sanity or distance check before stepping.
- Set joint axes and joint positions from the layout's physical drive axis and pivot. For a spool/anchor-chain style
  scene, the hinge axis should match the spool's physical shaft axis, and the joint position should be at the shaft
  center rather than at a flange or arbitrary bbox point.
- Drive external articulations through actuator/DOF controllers such as `control_dofs_position_velocity`, not by
  forcibly writing DOF velocity with `set_dofs_velocity` during the simulation.
- Cap PD gains and actuator force ranges in heavy IPC contact scenes. Overly strong controllers can cause large
  per-step joint-angle jumps that look like penetration and can destabilize contact.
- For diagnostics, do not trust `get_dofs_velocity()` alone on external articulations whose qpos is recovered from IPC.
  Estimate true angular velocity from step-to-step hinge angle differences when needed.
""".strip()


IPC_FAILURE_DIAGNOSTIC_GUIDE = """
IPC failure diagnostic guide:
- If stdout/stderr show libuipc/UIPC initial-geometry diagnostics such as `SimplicialSurfaceIntersectionCheck`,
  `SimplicialSurfaceDistanceCheck`, thickness/distance/barrier failures, or `World is not valid`, and the later Python
  exception says `IPC rigid state accessor feature is unavailable... requires rigid ABD state retrieval`, treat the
  accessor exception as a downstream/secondary diagnostic from the invalid IPC world. Do not infer from that pattern
  alone that the local libuipc build lacks rigid ABD accessor support.
- For that combined pattern, route the repair to the source that owns initial placement, scale, spacing, orientation,
  mesh clearance, duplicate IPC contact geometry, or generated asset topology. This is often `body`, but it may be a
  generated mesh/XML asset when the asset has a filled hole, wrong scale, extra component, inverted/invalid volume, or a
  shape that cannot physically interlock as requested. Preserve the intended IPC contact/coupling model; do not "fix"
  it by disabling IPC, setting `needs_coup=False`, changing the mechanism to hidden constraints, or bypassing contact
  unless a clean no-penetration repro still proves IPC capability is missing.
- Treat the rigid ABD accessor as an execution/libuipc capability issue only after a valid rigid IPC scene with no
  initial penetration, distance/thickness, or `World is not valid` diagnostics still fails to expose the accessor.
""".strip()
