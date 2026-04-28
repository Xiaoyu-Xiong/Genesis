from __future__ import annotations

from ...configs import CONFIGS


def build_parameter_notes() -> dict[str, str]:
    notes = {
        "scene.render.follow_entity.smoothing": (
            "Follow-camera smoothing factor. Higher values make camera motion smoother but increase lag."
        ),
        "scene.render.follow_entity.fixed_axis": (
            "Per-axis lock for follow-camera target position. Use null on an axis to follow the entity on that axis, "
            "or set a number to keep that axis fixed."
        ),
        "bodies[].collision.coup_restitution": (
            "Impact bounciness. Higher values create more rebound and usually make contact behavior less stable."
        ),
        "scene.ground_collision.friction": (
            "Contact friction coefficient. Higher values resist sliding more strongly, but do not guarantee perfectly non-slipping contact. Use 0.8 as a reasonable default."
        ),
        "bodies[].collision.friction": (
            "Contact friction coefficient for rigid bodies. Higher values resist sliding more strongly, but do not "
            "guarantee perfectly non-slipping contact. Use 0.8 as a reasonable default. Do not set this on FEM+IPC "
            "deformable bodies, where deformable contact friction is fixed by system defaults."
        ),
        "bodies[].rho": (
            "Material density. Higher rho makes the body heavier and increases inertia, but does not change geometric "
            "size. Keep density in the range 300 to 3000 kg/m^3."
        ),
        "bodies[].shape.default_armature": (
            "Additional articulated-joint armature used mainly for stability and numerical conditioning, not for "
            "task-level motion design."
        ),
        "bodies[].shape.scale": (
            "Uniform mesh scale factor. Use this when a mesh body's overall size is wrong for the scene: increase it "
            "to make the whole mesh larger, decrease it to make the whole mesh smaller. For deformable mesh bodies, "
            "this also changes the physical tetrahedralization size because the geometry itself is rescaled before "
            "remeshing and TetGen. When mesh bounding-box metadata is available, use that `bbox_size` evidence to "
            "estimate `shape.scale` instead of guessing."
        ),
        "bodies[].shape.file": (
            "For generated mesh bodies, the main runtime IR must point `shape.file` to the canonical repaired mesh "
            "path returned as `mesh_path` (typically under `processed/repaired*.obj`). Do not use `textured/model.obj` "
            "as the main runtime mesh, even when texture generation succeeds."
        ),
        "bodies[].fixed": (
            "Whether a rigid body is fixed in the world. Use this for rigid primitive, rigid mesh, or URDF obstacles, "
            "tables, and props that should not fall under gravity. For MJCF, express a fixed base in the XML itself."
        ),
        "bodies[].actuators[].kp": (
            "Position-control stiffness. Increasing kp makes tracking more aggressive, but if it is too large the "
            "joint can oscillate or destabilize."
        ),
        "bodies[].actuators[].kv": (
            "Position-control damping. kv suppresses oscillation and overshoot; too little damping can be shaky, "
            "too much can make motion sluggish."
        ),
        "bodies[].actuators[].force_range": (
            "Actuator output limit. This caps the maximum available force/torque; if it is too small, the joint may "
            "still be weak even when kp is large."
        ),
        "SetTargetPosActionIR.values": (
            "Target positions for position actuators. These are desired setpoints, not direct joint-state writes."
        ),
        "SetTorqueActionIR.values": (
            "Direct force/torque commands for motor actuators. These do not provide position tracking on their own."
        ),
        "bodies[].simulation_kind": (
            "Choose `deformable` when soft-body deformation visually makes the task closer to the prompt. Otherwise prefer `rigid`."
        ),
        "ApplyExternalWrenchActionIR.force": (
            "External force disturbance applied to a body or selected links. It is not an actuator command. Its "
            "effect persists across subsequent step actions until another wrench update changes it. If the effect is "
            "too weak or too strong, prefer adjusting force magnitude first before changing application duration."
        ),
        "ApplyExternalWrenchActionIR.torque": (
            "External torque disturbance applied to a body or selected links. It is not an actuator command. Its "
            "effect persists across subsequent step actions until another wrench update changes it."
        ),
        "ApplyExternalWrenchActionIR.ref": (
            "Reference point used for the external wrench (`link_origin`, `link_com`, or `root_com`). This changes "
            "how the same force produces translation versus rotation."
        ),
        "ApplyExternalWrenchActionIR.local": (
            "Whether the force/torque vector is interpreted in the world frame (`false`) or the target link's local "
            "frame (`true`)."
        ),
        "ObserveActionIR.entity": (
            "May be a single body name or a list of body names. Use the list form when observing several bodies with "
            "the same fields and tag at the same timestep."
        ),
        "SetPoseActionIR.entity": (
            "May be a single body name or a list of body names. Use the list form when broadcasting the same pose "
            "change to several bodies."
        ),
        "ApplyExternalWrenchActionIR.entity": (
            "May be a single body name or a list of body names. Use the list form when broadcasting the same "
            "external disturbance to several bodies."
        ),
    }
    if CONFIGS.deformable.simulation_backend == "pbd":
        notes["bodies[].deformable_material.stretch_compliance"] = (
            "PBD stretch compliance. Lower values make the soft body stiffer in edge-length preservation; higher "
            "values make it stretch and sag more easily. Very stiff elastic solids are often around 1e-8 to 1e-6, "
            "softer jelly-like solids around 1e-6 to 1e-4, and very floppy bodies can go higher. If the body is too "
            "floppy, decrease this value by about 3x to 10x; if it is too rigid, increase it by about 3x to 10x. "
            "Use **3e-5** as default for a moderately soft material with clearly visible deformation."
        )
        notes["bodies[].deformable_material.volume_compliance"] = (
            "PBD volume compliance. Lower values preserve volume more strongly; higher values allow more compression. "
            "If the body collapses or squashes too much, decrease this value; if it stays too incompressible, increase it. "
            "Use **3e-6** as default for visibly compressible but not completely mushy behavior."
        )
        notes["bodies[].deformable_material.rho"] = (
            "Density for PBD elastic bodies. Larger values make the soft body heavier without changing its geometry. "
            "Keep deformable density in the range 300 to 3000 kg/m^3."
        )
    else:
        notes["bodies[].deformable_material.E"] = (
            "Young's modulus for FEM elastic bodies, measured in Pascals, controls the body's resistance to stretching "
            "and compression. Higher `E` makes the body stiffer, lower `E` makes it softer. As a rough guide: very "
            "soft jelly-like solids are often around `1e4` to `5e4`, moderately soft rubbery solids around `5e4` to "
            "`5e5`, and firmer but still visibly deformable solids around `5e5` to `5e6`. If the body visibly collapses "
            "too much or cannot support load, increase `E` by about 3x to 10x; if it hardly deforms, decrease `E` by "
            "about 3x to 10x. Use about **1e5** as a good default initial guess for a medium-elastic soft solid."
        )
        notes["bodies[].deformable_material.nu"] = (
            "Poisson ratio for FEM elastic bodies. Higher values make the material less compressible. "
            "Use about **0.35** as a good default initial guess for a moderately compressible soft solid."
        )
        notes["bodies[].deformable_material.rho"] = (
            "Density for FEM elastic bodies. Larger values make the deformable body heavier without changing its "
            "geometry. Keep deformable density in the range 300 to 3000 kg/m^3."
        )
        notes["bodies[].initial_pose.pos"] = (
            "For FEM+IPC scenes, initial placements must avoid penetration and interpenetration. Leave a small positive "
            "clearance between deformable bodies, rigid bodies, and support surfaces instead of starting in overlap."
        )
    return notes


def build_parameter_relationship_notes() -> dict[str, str]:
    return {
        "position_actuator_tuning": (
            "For position actuators, kp sets how hard the controller tries to reach the target, kv damps motion, "
            "and force_range caps the actual output. If motion is too weak or the target is not reached, the cause "
            "may be insufficient kp, insufficient force_range, or both. If motion is too oscillatory, kp may be too "
            "high, kv may be too low, or force_range may be large enough to expose that instability. Critiques and "
            "fixes should distinguish between insufficient stiffness, insufficient damping, and insufficient output limit."
        ),
        "external_wrench_usage": (
            "`apply_external_wrench` is best understood as writing an external disturbance state into the solver, not "
            "as a one-step impulse helper. A common pattern is: set nonzero force/torque, step for some duration, then "
            "set the wrench back to zero. Critiques and fixes should distinguish between too-small wrench magnitude, "
            "too-short application duration, wrong reference point (`ref`), and wrong frame interpretation (`local`). "
            "When tuning the effect, prefer changing force/torque magnitude first and only then changing how long the "
            "wrench stays applied."
        ),
    }
