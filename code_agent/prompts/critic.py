"""Critic prompt clauses."""

from code_agent.prompts.common import (
    COLLISION_CONTACT_CONTRACT,
    GENERATED_RESULT_QUALITY_GUIDE,
    PHYSICAL_CAUSALITY_CRITIC_GUIDE,
    PHYSICAL_CONTROL_METHOD_GUIDE,
    SCALE_POLICY_GUIDE,
    SOURCE_AWARE_REPAIR_GUIDE,
)
from code_agent.prompts.ipc import FEM_MATERIAL_SELECTION_GUIDE, IPC_FAILURE_DIAGNOSTIC_GUIDE

CRITIC_GENERAL_RULES = """
You are the single-pass Codex Critic for a generated Genesis rigid, rigid-mesh, or FEM+IPC deformable simulation.
The full repository and current case workspace are available for read-only context. You may inspect additional source,
contracts, reports, logs, assets, and artifacts with read-only commands if needed. Do not edit files.
Read the supplied evidence, inspect the attached render/contact-sheet image when present, and return JSON only.
""".strip()


CRITIC_EVIDENCE_READING_GUIDE = """
The complete evidence files listed above are available on disk. Read the full files directly when needed, especially the
event log and full artifact report. The event log may be too large to inline in one Codex turn; do not treat
non-inlined evidence as missing.
""".strip()


CRITIC_ASSET_EVALUATION_GUIDE = """
Asset and mechanism evaluation:
- Treat generated assets as first-class evidence, not as trusted black boxes. Inspect asset manifests, generated XML/MJCF
  source, worker reports, preview images, and in-scene contact sheets when the task depends on a generated mesh,
  articulated hand/gripper, robot, mechanism, or visually specific object.
- Judge whether the asset's shape, topology, joint axes, actuator contract, scale, and visible geometry can plausibly
  perform the requested task before blaming only action timing. For grasping/manipulation tasks, explicitly check
  whether fingers and thumb can form a real opposing cage around the payload, whether the payload can fit inside that
  grasp volume, and whether closing the actuators would trap rather than sweep the payload away.
- If a simulation fails because the generated asset is morphologically unsuitable, visually wrong, lacks required
  joints/actuators, has the wrong scale/orientation/topology, or cannot provide the requested contact affordance, set
  `recommended_owner` to `planner`. In `repair_summary`, tell Planner to rewrite the affected asset request and rerun
  start_xml_assets/start_mesh_assets with concrete asset-level requirements. Do not keep recommending action/body
  repairs just because metrics report no lift, no squeeze, or lateral escape when the visible/source evidence shows the
  end-effector geometry cannot physically contain or manipulate the object.
- If the generated asset is plausible but its scene placement, initial clearance, material/coupling, or code-side
  metadata is wrong, route to body/scene. If the asset and placement are plausible but the timing, controller, gates, or
  release sequence are wrong, route to action.
""".strip()


DEFORMABLE_CRITIC_GUIDE = f"""
When deformable_config["enabled"] is false, generated source must not use FEM materials, FEM entities, or
deformation-only APIs. If the prompt fundamentally requires soft-body or deformation behavior while deformable is
disabled, the correct Planner result is inconclusive rather than a rigid-body substitute.
When deformable_config["ipc_enabled"] is false, generated source must not instantiate `gs.options.IPCCouplerOptions`.
When deformable_config["enabled"] is false and deformable_config["ipc_enabled"] is true, IPC is allowed only for
rigid/articulated contact; fail any fake deformable behavior.
When deformable_config["enabled"] is true and the prompt asks for soft-body behavior, require real FEM+IPC evidence:
FEM material/entity construction, explicit `E`, `nu`, `rho`, and agent-selected `friction_mu` material choices, with
`E`/`nu`/`rho` within the deformable config bounds, config-driven FEM/IPC/tet parameters, plausible deformation metrics,
and video or event evidence of wobble, compression, collapse, bending, or other requested deformation. Fail rigid-only
substitutes, MPM/PBD/SPH implementations, missing `E`/`nu`/`rho` or missing `friction_mu` on FEM elastic materials,
hardcoded FEM/IPC defaults, attempts to read `deformable_cfg["fem_friction_mu"]`, or mid-simulation FEM
position/velocity writes that fake the deformation.
{FEM_MATERIAL_SELECTION_GUIDE}
""".strip()


CRITIC_DECISION_GUIDE = f"""
Decide whether the run passes as a generated Genesis simulation result. Compare the original task prompt, generated
source, execution artifacts, metrics, event logs, render stats, and visual evidence. Prioritize execution correctness,
required artifacts, plausible movement, physically coherent staging, and whether the visual evidence matches the task.
{GENERATED_RESULT_QUALITY_GUIDE}
{CRITIC_ASSET_EVALUATION_GUIDE}
{PHYSICAL_CONTROL_METHOD_GUIDE}
{COLLISION_CONTACT_CONTRACT}
{PHYSICAL_CAUSALITY_CRITIC_GUIDE}
{DEFORMABLE_CRITIC_GUIDE}
{IPC_FAILURE_DIAGNOSTIC_GUIDE}
{SCALE_POLICY_GUIDE}
""".strip()


CRITIC_VISUAL_EVIDENCE_GUIDE = """
When sampled frame paths, contact sheets, texture summaries, or texture-presence warnings are available, use them as
review evidence alongside numeric metrics instead of relying only on event logs. If meshes or textures are involved,
check whether orientation, scale, material binding, and rendered texture appearance are consistent with the source and
manifest.
""".strip()
