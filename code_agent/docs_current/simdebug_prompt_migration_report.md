# SimDebug Prompt Migration Report

This report records the Milestone 1 audit and the Milestone 2 prompt-derived card coverage for the first Human
Experience Debug Cards skeleton. It intentionally does not remove text from the existing prompts yet. Prompt slimming
and live downstream dispatch are reserved for later milestones after card selection quality is easier to audit.

## Classification Rules

- `keep_in_prompt`: role identity, output protocol, schema requirements, command permissions, and harness action
  semantics that every invocation of that role needs.
- `api_context`: concrete Genesis API usage, function signatures, wrappers, and file-layout instructions. These should
  eventually be handled by the Genesis context pack or compact API reminders, not by debugging cards.
- `guideline_card`: diagnostic, routing, comparison, owner-selection, parameter-selection, or repair-direction advice.
- `restriction_card`: acceptance guards and self-checks that prevent physically invalid shortcuts or untrustworthy
  results.

## Prompt-Derived Cards

The card library is organized by operational use, not by migration status:

```text
code_agent/context/simdebug/
  catalog.json
  guideline_cards/
    assets/
    ipc/
    opt/
    repair_routing/
  restriction_cards/
    assets/
    evidence/
    opt/
    physics_validity/
```

The generated catalog is `code_agent/context/simdebug/catalog.json`. It currently contains 25 prompt-derived cards.

| Prompt source | Classification | Prompt-derived card |
| --- | --- | --- |
| `common.PHYSICAL_CAUSALITY_CONTRACT` | `restriction_card` | `physical_causality_restriction` |
| `common.PHYSICAL_CAUSALITY_CRITIC_GUIDE` | `restriction_card` | `physical_causality_restriction` |
| `common.PHYSICAL_CONTROL_METHOD_GUIDE` | `restriction_card` | `physical_causality_restriction` |
| `common.COLLISION_CONTACT_CONTRACT` | `restriction_card` | `collision_contact_restriction` |
| `common.SCALE_POLICY_GUIDE` | `restriction_card` | `scale_policy_restriction` |
| `common.BUILTIN_ASSET_POLICY_GUIDE` | `restriction_card` | `builtin_asset_policy_restriction` |
| `common.RENDER_CLARITY_GUIDE` | `restriction_card` | `render_visual_evidence_restriction` |
| `common.GENERATED_RESULT_QUALITY_GUIDE` | `restriction_card` | `render_visual_evidence_restriction` plus role-level prompt protocol |
| `common.SOURCE_AWARE_REPAIR_GUIDE` | `guideline_card` | `source_aware_repair_guideline` |
| `ipc.FEM_MATERIAL_SELECTION_GUIDE` | `guideline_card` | `ipc_fem_material_selection_guideline` |
| `ipc.RIGID_IPC_COUPLING_GUIDE` | `guideline_card` | `rigid_ipc_coupling_guideline` |
| `ipc.EXTERNAL_ARTICULATION_MJCF_GUIDE` | `guideline_card` | `external_articulation_mjcf_guideline` |
| `ipc.IPC_FAILURE_DIAGNOSTIC_GUIDE` | `guideline_card` | `ipc_initial_geometry_failure_diagnosis_guideline` |
| `critic.CRITIC_ASSET_EVALUATION_GUIDE` | `guideline_card` | `critic_asset_evaluation_guideline` |
| `critic.DEFORMABLE_CRITIC_GUIDE` | `restriction_card` | `deformable_fem_ipc_scope_restriction` plus `ipc_fem_material_selection_guideline` |
| `critic.CRITIC_VISUAL_EVIDENCE_GUIDE` | `restriction_card` | `render_visual_evidence_restriction` |
| `worker.RIGID_API_GUIDE` generated mesh manifest clauses | `guideline_card` | `generated_mesh_manifest_usage_guideline` |
| `worker.RIGID_API_GUIDE` rigid IPC config/control clauses | `guideline_card` / `restriction_card` | `ipc_runtime_config_mapping_guideline`, `rigid_ipc_coupling_guideline`, `physical_causality_restriction` |
| `worker.FEM_IPC_API_GUIDE` FEM/IPC config mapping clauses | `guideline_card` | `ipc_runtime_config_mapping_guideline` |
| `worker.FEM_IPC_API_GUIDE` duplicate plane and FEM initial-clearance clauses | `restriction_card` | `fem_ipc_initial_geometry_restriction` |
| `worker.FEM_IPC_API_GUIDE` FEM state/metrics clauses | `guideline_card` | `fem_state_metrics_guideline` |
| `planner.PLANNER_ACTION_POLICY_GUIDE` Opt-routing clauses | `guideline_card` | `opt_routing_guideline` |
| `planner.PLANNER_ACTION_POLICY_GUIDE` IPC invalid-world clauses | `guideline_card` | `ipc_initial_geometry_failure_diagnosis_guideline` |
| `planner.planner_available_actions_section` generated mesh retry and metadata clauses | `guideline_card` | `planner_asset_retry_guideline` |
| `planner.planner_available_actions_section` XML/MJCF asset request and actuator-contract clauses | `guideline_card` | `xml_asset_request_contract_guideline` |
| `planner.planner_available_actions_section` texture-dependent asset request clauses | `guideline_card` | `visual_texture_asset_request_guideline` |
| `planner.planner_fem_ipc_capability_section` adaptive IPC contact-distance clauses | `guideline_card` | `ipc_runtime_config_mapping_guideline` |
| `opt.build_opt_prompt` parameter-effectiveness clauses | `restriction_card` | `opt_effective_parameter_restriction` |
| `opt.build_opt_prompt` Opt candidate/structural-failure clauses | `guideline_card` | `opt_routing_guideline` |
| `opt.build_opt_prompt` metric/objective/success-criteria clauses | `guideline_card` | `opt_metric_and_objective_design_guideline` |
| `opt.build_opt_prompt` search-space and strategy clauses | `guideline_card` | `opt_search_space_design_guideline` |
| `opt.build_opt_prompt` XML scalar patch clauses | `restriction_card` | `opt_xml_scalar_patch_restriction` |
| `opt.build_opt_prompt` task-semantics preservation clauses | `restriction_card` | `opt_task_semantics_restriction` |

## Kept In Prompt For Now

These sections are not represented as cards in Milestone 2 because they are primarily role protocol, schema/action
semantics, or API context rather than reusable debugging experience:

- `planner.PLANNER_GENERAL_RULES`: Planner identity, JSON action requirement, inspection permissions.
- `planner.planner_available_actions_section`: action schema semantics, asset action contracts, command names, timing
  defaults, and harness workflow. Several embedded debugging clauses now have prompt-derived cards, but the action list itself
  remains prompt scaffolding.
- `planner.planner_fem_ipc_capability_section`: capability booleans, effective config path, and config values. The
  debugging parts are covered by FEM/IPC cards.
- `critic.CRITIC_GENERAL_RULES`: Critic identity, read-only behavior, JSON-only response.
- `critic.CRITIC_EVIDENCE_READING_GUIDE`: evidence-reading protocol.
- `critic.CRITIC_DECISION_GUIDE`: high-level decision wrapper. Embedded checks are now represented by cards.
- `worker.WORKER_COMMON_RULES`: module ownership, edit boundaries, command permissions, and execution ban.
- `worker.RIGID_API_GUIDE`: concrete Genesis rigid API and generated-mesh instantiation details, plus embedded IPC cards.
- `worker.FEM_IPC_API_GUIDE`: concrete FEM/IPC API and config mapping details, plus embedded FEM/IPC cards.
- `opt.build_opt_prompt`: Opt role protocol, required final schema, command paths, and implementation rules. The
  prompt-derived cards cover candidate routing, physical restrictions, visual evidence, and effective-parameter checks.

## Current Runtime Behavior

- Planner loads cards through `code_agent.context.simdebug`.
- Python only filters card candidates by declared target role and active physics mode.
- Planner is responsible for judging task/evidence relevance and dispatching every useful card; there is no fixed top-k
  cap and no hard-coded generic-tag or stopword filter.
- Planner writes the latest candidate bundle to `reports/simdebug_card_dispatch.json` and appends each prompt-build selection to
  `reports/simdebug_card_dispatch.jsonl`.
- Downstream prompt slimming is not enabled yet. The current implementation is an auditable skeleton that lets us test
  dispatch quality before removing prompt-derived clauses from static prompts.
