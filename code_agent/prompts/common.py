"""Shared prompt clauses used by Planner, Writer, XML asset, and Critic calls."""

PHYSICAL_CAUSALITY_CONTRACT = """
Physical causality contract:
- During simulation, every non-static object's motion must be explainable by the simulator's physical dynamics and by
  explicitly modeled control channels.
- After initialization, do not move task objects by direct pose/qpos/qvel writes, hidden constraints, target-following
  proxy forces, or outcome-directed external forces. A task object may move because of gravity, contact, friction,
  collision, joints, actuators, or an explicitly requested physical field/effect.
- If an object is manipulated by a robot or mechanism, the robot/mechanism should act through its actuators, joints,
  contacts, friction, or collision geometry. Do not servo the manipulated object itself toward an end-effector, handoff
  point, bin, platform, or goal pose.
- External forces are allowed only when they are part of the requested scene physics or a clearly modeled mechanism,
  such as wind, thrust, magnetism, springs, launch impulse, or actuator force. They must be applied to the physically
  appropriate body, recorded in metrics, and must not directly encode the desired task outcome.
- When a task requires an attachment-like effect, model it explicitly through geometry, joints, contacts, actuator
  force, or a task-requested physical mechanism. If that is not possible with the current assets/contracts, fail clearly
  and route repair to the source owner instead of hiding the gap with object-following assistance.
""".strip()


SCALE_POLICY_GUIDE = """
Scale policy:
- Avoid non-uniform scale by default. Unless the input task prompt explicitly requests an exceptional anisotropic
  scaling operation, do not use per-axis scale values such as `scale=(sx, sy, sz)` with unequal components for meshes,
  primitives, imported assets, MJCF/XML assets, or generated asset manifest requests.
- When an object needs a long, flat, thick, thin, or otherwise non-cubic shape, model that shape directly with primitive
  dimensions (`size`, `radius`, `height`), generated-asset geometry, XML primitive geom sizes, or a regenerated asset
  with the intended proportions. Do not create the intended proportions by stretching an already generated/imported
  mesh with non-uniform scale.
- Uniform scalar scale is acceptable for unit conversion, global sizing, and layout fitting. If a rare task genuinely
  requires non-uniform scale, document the reason in the plan or worker report and keep it isolated to that explicit
  requirement.
""".strip()


BUILTIN_ASSET_POLICY_GUIDE = """
Built-in Genesis asset policy:
- Do not inspect, copy, import, or reference prepackaged assets under `genesis/assets`.
- Do not use `gs.utils.get_assets_dir()`, `genesis.utils.misc.get_assets_dir()`, or Genesis built-in relative asset
  paths such as `xml/...`, `urdf/...`, or `meshes/...` for task geometry, robots, textures, or props. The only
  exception is an XML/MJCF file referencing a mesh file that was generated into the same case workspace and validated
  by the XML asset pipeline.
- Use Genesis primitive morphs, case-workspace generated XML/MJCF assets, Meshy-generated assets, or explicit
  user-provided layout assets that have been copied into the case workspace. XML/MJCF mesh references are allowed only
  when the mesh files are generated case-workspace assets validated by the XML asset pipeline.
- This is enforced by the Codex invocation sandbox and by static validation of planner outputs and generated source.
""".strip()


PHYSICAL_CAUSALITY_CRITIC_GUIDE = """
Audit physical causality. Identify which entities are controlled, which entities are passive task objects, and which
APIs modify them during simulation. Fail the result if a passive task object reaches the goal mainly through direct
state writes, hidden attachment, proxy constraints, or external forces that track an end-effector/goal pose instead of
arising from modeled contact, joints, actuators, or an explicitly requested physical effect. Treat numeric success as
insufficient when the source or video shows object-following assistance that bypasses the requested physical mechanism.
""".strip()


PHYSICAL_CONTROL_METHOD_GUIDE = f"""
Physical plausibility includes the control method, not only the final video. Direct state writes such as setting root
pose, qpos, entity pose, DOF position, or velocity are acceptable for initialization only. After stepping begins,
movement should be driven by actuator commands, DOF controllers, motors, external forces/torques, or pre-step initial
velocities. For XML/MJCF articulated assets, expected repairs should preserve the designed actuator/DOF control path or
request a body/asset contract fix that exposes a controllable joint; do not recommend mid-simulation pose
teleportation as a repair for locomotion or mechanism motion.
{PHYSICAL_CAUSALITY_CONTRACT}
""".strip()


RENDER_CLARITY_GUIDE = """
Rendering clarity requirement:
- Generated videos must clearly show the task-relevant scene, objects, contacts, and motion. Camera position, lookat,
  field of view, clipping planes, lighting, background, capture cadence, and resolution should be chosen so the whole
  requested behavior is readable without severe cropping, tiny objects, occlusion, blank frames, or confusing views.
- If critic or visual evidence says the result is hard to inspect because the camera is too far, too close, poorly
  aimed, poorly lit, cropped, static when tracking is needed, or otherwise unclear, actively repair the rendering module
  and camera parameters. Do not treat unclear rendering as acceptable when the simulation behavior cannot be judged.
- Record final camera parameters and rendering choices in render_stats.json so repairs can be source-aware.
""".strip()


GENERATED_RESULT_QUALITY_GUIDE = f"""
The final simulation should not merely satisfy numeric proxies. It should match the input text prompt and look
physically and visually reasonable, coherent, and logically staged.
{RENDER_CLARITY_GUIDE}
""".strip()


SOURCE_AWARE_REPAIR_GUIDE = """
Repair guidance must be detailed, source-aware, and directly actionable. Compare the original text prompt, latest
execution and visual output, metrics/event logs, relevant generated source, and module ownership. Name the owner module,
the concrete behavior that is wrong, the evidence proving it, the likely source-level cause, and what a convincing fix
should accomplish. Avoid vague advice like "improve the trajectory" when concrete evidence exists. Do not compress
important source-level feedback just to keep the answer short. If the evidence points to a generated mesh asset itself
being invalid or unsuitable (failed manifold check, Genesis FEM import validation failure, missing/corrupt texture,
wrong generated topology, or a visual/runtime mesh pairing defect), route the fix to Planner/mesh asset regeneration
instead of asking scene/body/action/rendering workers to patch, reshape, retopologize, or procedurally replace that mesh.
If the evidence points to a generated XML/MJCF/URDF asset itself being unsuitable for the requested mechanism
(for example a gripper without a real opposing thumb, fingers that cannot form an enclosing cage, missing useful
actuator affordances, wrong joint axes, invalid link hierarchy, or body geometry that cannot perform the requested
contact task), route the fix to Planner/XML asset regeneration. Do not keep assigning action/body repairs when the
action is only failing because the generated articulated asset cannot physically do the job.
""".strip()
