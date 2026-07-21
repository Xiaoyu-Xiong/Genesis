"""Critic prompt clauses after SimDebug card migration."""

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


CRITIC_DECISION_GUIDE = """
Decide whether the run passes as a generated Genesis simulation result. Compare the original task prompt, generated
source, execution artifacts, metrics, event logs, render stats, visual evidence, generated assets, and Planner-dispatched
SimDebug cards. Prioritize execution correctness, required artifacts, plausible movement, physically coherent staging,
task match, and readable evidence.
When `failure_class=execution.insufficient_frame_progress`, require source-aware rework rather than recommending a
larger timeout or a blind retry. Inspect fresh partial frames, frame timing, geometry complexity, solver/contact behavior,
and render cost, then route the repair to scene, body, action, or rendering according to the actual bottleneck.
""".strip()


CRITIC_ASSET_EVALUATION_GUIDE = """
Use the Planner-dispatched SimDebug cards for asset and mechanism evaluation guidance.
""".strip()


DEFORMABLE_CRITIC_GUIDE = """
Use the Planner-dispatched SimDebug cards for FEM/IPC scope and material-evidence checks.
""".strip()


CRITIC_VISUAL_EVIDENCE_GUIDE = """
Use the Planner-dispatched SimDebug cards for visual, texture, and manifest evidence checks.
""".strip()
