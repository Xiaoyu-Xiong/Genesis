"""Compact FEM, IPC, and rigid-coupling prompt hooks.

The detailed FEM/IPC debugging and restriction content is represented as
Planner-dispatched SimDebug cards. See `code_agent/prompts_legacy/ipc.py` for
the pre-card prompt text.
"""

IPC_SIMDEBUG_GUIDE = """
Use the Planner-dispatched SimDebug cards for FEM material choices, rigid IPC coupling modes, external-articulation
MJCF/XML setup, IPC runtime config mapping, FEM evidence, and IPC initial-geometry failure diagnosis.
""".strip()


FEM_MATERIAL_SELECTION_GUIDE = IPC_SIMDEBUG_GUIDE
RIGID_IPC_COUPLING_GUIDE = IPC_SIMDEBUG_GUIDE
EXTERNAL_ARTICULATION_MJCF_GUIDE = IPC_SIMDEBUG_GUIDE
IPC_FAILURE_DIAGNOSTIC_GUIDE = IPC_SIMDEBUG_GUIDE
