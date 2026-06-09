"""Shared compact prompt hooks.

Detailed simulation debugging guidance now lives in
`code_agent/context/simdebug/` and is dispatched by Planner as role-specific
cards. The legacy pre-card prompt text is kept under `code_agent/prompts_legacy/`.
"""

SIMDEBUG_CARD_GUIDANCE_HOOK = """
Follow the Planner-dispatched SimDebug cards for simulation debugging guidance, physical validity restrictions,
asset-use rules, visual-evidence requirements, source-aware repair routing, and optimization guardrails. Treat those
cards as the active source of task-specific debugging experience for this invocation.
""".strip()


PHYSICAL_CAUSALITY_CONTRACT = SIMDEBUG_CARD_GUIDANCE_HOOK
COLLISION_CONTACT_CONTRACT = SIMDEBUG_CARD_GUIDANCE_HOOK
SCALE_POLICY_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
BUILTIN_ASSET_POLICY_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
PHYSICAL_CAUSALITY_CRITIC_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
PHYSICAL_CONTROL_METHOD_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
RENDER_CLARITY_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
GENERATED_RESULT_QUALITY_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
SOURCE_AWARE_REPAIR_GUIDE = SIMDEBUG_CARD_GUIDANCE_HOOK
