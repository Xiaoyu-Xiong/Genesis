# Human Experience Debug Cards Plan

This document proposes a practical mechanism for moving simulation-debugging experience out of the static prompt layer
and into a Planner-orchestrated card system. The current `code_agent/prompts/` package already contains many useful
domain guides and failure-diagnosis rules. The plan is to keep only general role, protocol, and schema instructions in
the prompts, extract the more situational guidance into structured debug cards, and let **Planner** decide which cards
each downstream agent should see.

The core idea is to store human and prompt-derived experience as one unified structured debug-card library. A card may
contain constructive `guidance`, guardrail-style `restrictions`, `checks`, and `dispatch_hints` together. Planner is the
only agent that directly reads and selects cards. Downstream agents receive a compact Planner-dispatched card bundle
tailored to their current role, task state, and latest evidence.

## Motivation

The current pipeline already injects Genesis API context and local documentation. That helps agents call the right APIs,
but it does not fully teach them how experienced simulation developers debug scenes, distinguish structural failures
from tunable residuals, or avoid physically invalid shortcuts.

At the same time, many existing prompt clauses are already doing this job manually. Examples include FEM material
selection, IPC coupling-mode advice, IPC failure diagnosis, physical-causality restrictions, source-aware repair
guidance, Opt variable-selection rules, and visual-evidence requirements. These clauses are valuable, but keeping them
inside role prompts has drawbacks:

- Every agent sees broad guidance even when only a small part is relevant.
- Prompt files grow into mixed collections of role protocol, API reminders, failure taxonomies, and repair heuristics.
- It is hard to audit which specific guidance affected a case.
- It is hard to update, disable, or ablate one piece of guidance without editing large prompt strings.

The proposed refactor treats `prompts/` as the first source of high-quality prompt-derived cards. Prompt files should
become mostly general scaffolding, while situational debugging knowledge moves into a Planner-dispatched card library.

The missing layer is not more raw simulation code. The missing layer is a compact, searchable, and executable form of
expert experience:

- What symptoms imply which likely causes.
- Which variables are worth changing under a specific failure mode.
- Which comparisons reveal whether one rollout is better than another.
- Which shortcuts must be rejected even if metrics look good.
- Which probes should be run before asking a writer or Opt to repair a case.

The proposed solution is a **Human Experience Debug Card Library** that sits beside the existing Genesis context pack.
Each episode gives Planner access to API documentation and the debug-card catalog. Planner then distributes relevant
card summaries to writers, Critic, and Opt as part of its normal global-context orchestration.

## Related Work Signals

### Retrieval-Augmented Knowledge Use

RAG, DocPrompting, and Gorilla all show that model outputs improve when the model retrieves targeted external knowledge
before generating an answer or code. In these systems, the model does not rely only on memorized parameters. It retrieves
documents, API descriptions, or examples, then conditions generation on the retrieved context.

For `code_agent`, this pattern is already partially present through Genesis API/context injection. The extension proposed
here is to let Planner retrieve **human debugging experience**, not only API usage, and then decide what to distribute
to each downstream agent. API context answers "what functions exist"; debug cards answer "what tends to go wrong, how to
diagnose it, and what repair direction is physically valid."

### Human Feedback As Comparisons And Directional Corrections

TAMER and COACH-style human-feedback work shows that human input is often more useful when it is local, evaluative, and
directional than when it is a broad natural-language essay. The human does not need to write a full policy or complete
program. They can say that one behavior is better than another, or that the current action should be encouraged or
discouraged.

For `code_agent`, this suggests that human experience cards should include comparison criteria and repair direction:

- "If both rollouts pass a binary metric, prefer the one with visible contact-driven motion."
- "If increasing plunger travel improves alignment but destroys rebound, tune phase/hold before further increasing
  travel."
- "If a parameter perturbation does not change reported effective controls, repair the hook before running Opt."

These are stronger than generic instructions because they tell the agent how to rank alternatives and how to move after
observing a symptom.

### Constraints, Counterexamples, And Self-Checks

CEGIS and Sketch show that human knowledge can be passed as specifications, holes, and counterexamples rather than full
solutions. Constitutional AI shows a related pattern for LLMs: high-level principles can be used as an automatic
self-critique and revision mechanism.

For `code_agent`, this motivates putting restriction/check sections inside cards. These sections are not optimization
hints; they are acceptance guards. They should help agents reject invalid solutions before finalizing:

- Do not move passive task objects through direct post-initialization pose/qpos/qvel writes.
- Do not add hidden springs, constraints, suction, or target-following proxy forces unless the prompt explicitly asks.
- Do not treat camera/rendering edits as Opt variables.
- Do not call a case successful if active Opt variables are clamped away or missing from metrics.
- Do not accept numeric success without readable visual evidence for contact-rich tasks.

Card checks should also produce counterexample-style questions: "Would this still pass if the hidden state write is
removed?" or "Does a small perturbation of the exposed parameter change the measured behavior?"

## Unified Card Content

The library should not split cards into guideline cards and restriction cards. In practice, the same useful experience
often contains both a positive recommendation and a negative guardrail. For example, a collision card may say to keep
independent object collision enabled by default, and also say not to disable contact masks for task-critical pairs.

A unified card should be used when the agent needs one or more of these:

- Which owner module likely needs repair.
- Which quantities to measure.
- Which parameters are good Opt variables.
- Which optimizer strategy is appropriate.
- Which rollout is better under a subtle physical tradeoff.
- Whether a generated behavior is physically causal.
- Whether metrics can be trusted.
- Whether a result should be routed to repair instead of accepted.
- Whether an Opt run is meaningful.

Example unified card:

```yaml
schema_version: 1
id: fem_dual_plunger_leveling
title: FEM dual-plunger bridge leveling
scopes: [planner, action, opt, critic]
physics_modes: [fem_ipc]
task_tags: [precision_control, deformable, leveling, dual_actuator]
failure_tags:
  - asymmetric_alignment
  - overcompression
guidance:
  - left and right tick heights do not align at the same time
  - one column remains high while the other over-compresses
  - bridge rocks past level after plunger release
  - prefer lower simultaneous left/right residual over a single-side minimum
  - prefer contact-driven alignment with visible column deformation over direct bridge control
checks:
  - check left/right tick height residuals over time
  - check bridge roll angle at the best alignment time
  - check column compression and rebound after release
dispatch_hints:
  - send to action when controller timing or force staging is being repaired
  - send to opt when exposed scalar variables can tune a runnable but imperfect result
provenance:
  source_type: human_authored
```

Cards can still contain specialized fields when useful, but `guidance`, `restrictions`, `checks`, and
`dispatch_hints` should be the common vocabulary.

Optional card fields for optimization-heavy cards:

```yaml
recommended_variables:
  - action.left_plunger_travel_m
  - action.right_plunger_travel_m
  - action.phase_delay_s
  - action.hold_duration_s
  - body.column_youngs_modulus_pa
recommended_metrics:
  - left_tick_height_error_m
  - right_tick_height_error_m
  - simultaneous_alignment_window_s
  - bridge_roll_angle_rad
  - left_rebound_fraction
  - right_rebound_fraction
```

## Card Schema

A minimal card schema should support Planner-owned retrieval and role-specific card dispatch.

Recommended fields:

```yaml
schema_version: 1
id: string
title: string
summary: string
scopes:
  - planner
  - scene
  - body
  - action
  - rendering
  - critic
  - opt
physics_modes:
  - rigid
  - rigid_ipc
  - fem_ipc
task_tags:
  - string
failure_tags:
  - string
guidance:
  - string
restrictions:
  - string
checks:
  - string
recommended_variables:
  - string
recommended_metrics:
  - string
comparison_rules:
  - string
safe_fixes:
  - string
failure_routing:
  - string
dispatch_hints:
  - string
source_notes:
  - string
provenance:
  source_type: prompt_migration | human_authored | case_distilled
  source_path: string
  source_symbol: string
  created_by: string
  created_at: string
  validated_on_cases:
    - string
  confidence: low | medium | high
```

Not every field is required for every card, but the Planner retrieval layer should require `id`, `title`, `summary`,
`scopes`, and `physics_modes`. The `scopes` field is the source of truth for every role Planner may dispatch the card
to.

## Prompt Decomposition And Card Extraction

The first source of cards should be the existing prompt library. This is both lower effort and safer than starting from
a blank card set, because the current prompts already encode many hard-won rules from previous pipeline iterations.

The migration should classify each prompt clause into one of four buckets:

1. **Keep in prompts**
   - Role identity and responsibilities.
   - Input/output schemas and JSON response contracts.
   - Allowed tools, file ownership, and edit boundaries.
   - Current case state, request payloads, and workspace summaries.
   - Short universal reminders that must always be present.
2. **Move to SimDebug cards as guidance**
   - Diagnostic heuristics.
   - Suggested variables, metrics, and optimization strategies.
   - Asset, IPC, FEM, rendering, and source-aware repair advice that is only relevant for some cases.
   - Comparison rules for choosing between imperfect rollouts.
3. **Move to SimDebug cards as restrictions/checks**
   - Forbidden physical shortcuts.
   - Evidence requirements for accepting success.
   - Rules that reject invalid Opt, contact, asset, or rendering outcomes.
   - Counterexample-style checks.
4. **Move to API/context docs or leave in Genesis context**
   - Pure API syntax.
   - Stable Genesis reference material.
   - Local source snippets that are better represented by the existing Genesis context pack.

### Candidate Prompt Sources

The current prompt files contain several high-value migration targets:

- `code_agent/prompts/common.py`
  - `PHYSICAL_CAUSALITY_CONTRACT` can become action/critic cards with hard restriction sections, while prompts keep only a short
    universal causality reminder.
  - `COLLISION_CONTACT_CONTRACT` can become body/contact cards.
  - `SCALE_POLICY_GUIDE` can become scene/body/rendering cards with scale guidance and checks.
  - `RENDER_CLARITY_GUIDE` can become rendering-evidence cards dispatched mostly to rendering and Critic.
  - `SOURCE_AWARE_REPAIR_GUIDE` can become Planner/Critic repair-routing cards.
- `code_agent/prompts/ipc.py`
  - `FEM_MATERIAL_SELECTION_GUIDE` should become FEM material cards.
  - `RIGID_IPC_COUPLING_GUIDE` should become rigid-IPC coupling selection cards.
  - `EXTERNAL_ARTICULATION_MJCF_GUIDE` should become XML/MJCF external-articulation cards.
  - `IPC_FAILURE_DIAGNOSTIC_GUIDE` should become IPC failure-diagnosis and repair-routing cards.
- `code_agent/prompts/opt.py`
  - Stable Opt role protocol, output schema, and execution commands should stay in the prompt.
  - Variable-selection heuristics, parameter-effectiveness checks, visual-verification requirements, log-scale advice,
    and "do not optimize rendering" rules should move into Opt-owned cards.
- `code_agent/prompts/planner.py`
  - Planner action schemas and action availability should stay in the prompt.
  - Long policy clauses for asset failure routing, geometry failure diagnosis, Opt candidacy, and repair routing should
    become Planner-dispatched cards.
- `code_agent/prompts/critic.py`
  - Critic response schema and read-only role should stay in the prompt.
  - Asset evaluation, deformable evidence requirements, visual evidence checks, and IPC failure interpretation should
    become Critic/rendering/body cards.
- `code_agent/prompts/worker.py`
  - Worker file ownership, target-file editing rules, and module export contracts should stay in the prompt.
  - FEM/IPC placement pitfalls, duplicated support geometry, post-step FEM state-write warnings, and owner-specific
    implementation heuristics should become cards dispatched by Planner to the relevant writer.

### Prompt Slimming Rule

After migration, prompts should mainly answer:

- Who is this agent?
- What files or actions can it touch?
- What input has it received?
- What output schema must it return?
- What compact card bundle did Planner dispatch for this call?

Prompts should not remain the primary storage location for long lists of simulation heuristics. Detailed domain
experience should live in cards so it can be selected, audited, ablated, and revised independently.

### Migration Workflow

Prompt-to-card migration should be explicit and reviewable:

1. Inventory prompt constants and long policy sections.
2. Split each section into atomic rules or heuristics.
3. Classify each atom as `keep_in_prompt`, `simdebug_card`, or `api_context`.
4. Create YAML cards with `provenance.source_type=prompt_migration`, `source_path`, and `source_symbol`.
5. Replace the prompt-derived text with a short general instruction that Planner may dispatch relevant cards.
6. Run prompt-size and behavior diffs on representative suites.
7. Keep Planner dispatch auditable so prompt-derived cards can be evaluated before the matching static prompt text is
   removed.

## Library Layout

Start with a file-based library before adding a database.

```text
code_agent/context/simdebug/
  catalog.json
  cards/
    planner/
      planner_card_dispatch_guideline.yaml
      planner_asset_retry_guideline.yaml
      xml_asset_request_contract_guideline.yaml
    scene/
      ipc_runtime_config_mapping_guideline.yaml
      scale_policy_restriction.yaml
      soft_body_robust_layout_guideline.yaml
    body/
      generated_mesh_manifest_usage_guideline.yaml
      collision_contact_restriction.yaml
      rigid_ipc_coupling_guideline.yaml
    action/
      controller_schedule_guideline.yaml
      physical_causality_restriction.yaml
      rigid_contact_metrics_guideline.yaml
    rendering/
      render_visual_evidence_restriction.yaml
    critic/
      critic_asset_evaluation_guideline.yaml
      source_aware_repair_guideline.yaml
    opt/
      opt_effective_parameter_restriction.yaml
      opt_metric_and_objective_design_guideline.yaml
```

Do not add a separate README under `context/simdebug/`. Operational documentation for both Genesis context and
simdebug cards should live in `code_agent/docs/context.md`.

`catalog.json` should be generated from the YAML cards and include the fields needed for retrieval:

- `id`
- `title`
- `summary`
- `scopes`
- `physics_modes`
- `task_tags`
- `failure_tags`
- source path and source symbol for prompt-derived cards

The subdirectory under `cards/` is the primary agent owner for maintainability. It is not the complete dispatch set;
the card's `scopes` field remains the complete list of roles Planner may send that card to.

## Planner-Owned Selection And Dispatch

The card selector should be deterministic at first. A simple hybrid strategy is enough, but it should run through
Planner rather than through each downstream agent independently.

Planner builds a global case state from:

- original prompt
- effective physics mode and deformable/IPC config
- planner output and dispatch graph
- generated source summaries when available
- asset generation and inspection reports
- latest execution stderr/stdout digest
- metrics keys and failure values
- visual/critic evidence
- Opt diagnosis, exposed variables, and trial reports
- prior cards already dispatched in the current episode

Planner then performs two related decisions:

1. **Global card selection**
   - Hard filter by physics mode, enabled capabilities, and broad task tags.
   - Score lexical overlap against prompt, reports, metrics, logs, and failure summaries.
   - Prefer high-confidence cards validated on similar cases.
   - Keep all cards Planner judges relevant to the current case state. Do not impose a fixed top-k limit at this stage.
2. **Role-specific dispatch**
   - For each downstream call, filter the global working set by `scopes`.
   - Add or remove cards based on the current Planner action.
   - Convert selected cards into a compact role-specific section.
   - Record the dispatched card IDs, reasons, and target role in the case report.

The important design rule is: downstream agents should not open the card library themselves. They receive only the card
bundle Planner chooses to distribute. This preserves Planner's global context advantage and keeps file I/O centralized.

Planner may dispatch different card bundles at different times:

- Before `write_plan`, Planner may use structure and success-metric guidelines.
- Before spawning writers, Planner may give each owner only the relevant implementation cards.
- After execution failure, Planner may switch to symptom-specific diagnostic cards.
- Before Critic, Planner may emphasize evidence, causality, and source-routing card sections.
- Before Opt, Planner may emphasize variable, objective, and parameter-effectiveness cards.
- After Opt returns, Planner may update the Critic card bundle to include Opt-specific evidence checks.

The dispatch format should be compact. Do not paste full cards into every prompt. Planner may select many relevant
cards, but it should summarize and group them into a readable role-specific section rather than forwarding raw YAML.
Convert selected cards into a short
"Planner-dispatched human debugging experience" section:

```text
Planner-dispatched human debugging experience:
- [card:fem_dual_plunger_leveling] For dual FEM plunger leveling, measure simultaneous left/right tick
  height residuals, bridge roll, and rebound. Prefer tuning left/right travel and phase delay before material changes.
- [card:passive_object_no_direct_state_write] Reject success if the bridge or passive objects are moved by
  direct post-initialization state writes or hidden constraints.
```

## Pipeline Integration Points

### MVP-First Rollout Strategy

This plan should be implemented as a lightweight framework first, then improved by adding better cards over time. The
initial engineering goal is not to build a complete expert library. The initial goal is to make debug cards a real part
of the pipeline:

```text
card files -> catalog -> Planner selection -> Planner dispatch -> role-specific prompt section -> dispatch audit record
```

The first version can be useful as soon as the plumbing is stable and auditable, even before every card is polished.
The framework should support all prompt-derived cards from the beginning, while allowing Planner to decide which
ones are relevant for each task and downstream role.

The minimal viable integration should include:

- A prompt audit that classifies existing prompt clauses into `keep_in_prompt`, `simdebug_card`, or `api_context`.
- A YAML card directory and schema validator.
- A generated `catalog.json`.
- A deterministic Planner-side selector that can return zero or more cards for the current case state.
- Slimmed prompt builders that keep role protocol and schema instructions while accepting Planner-dispatched cards.
- Planner action payloads or prompt briefs that can carry compact role-specific card bundles.
- Prompt builders for writer, Critic, and Opt calls that render only the cards Planner dispatched to that role.
- A report field recording selected card IDs, target roles, and dispatch reasons for each Planner turn and
  downstream call.
- A config flag to disable card dispatch globally or per suite.

The MVP should avoid expensive probes, embeddings, fine-tuning, or automated card distillation. Those can be added after
the framework has proven that Planner-selected cards help the existing Planner/Writer/Critic/Opt loop.

Recommended first connection order:

1. **Prompt audit and prompt-derived card skeleton**
   - Classify existing prompt clauses.
   - Move all clauses classified as situational guidance into disabled or dry-run cards.
   - Keep prompts behaviorally unchanged until dispatch is ready.
2. **Planner-only visibility and audit**
   - Planner can see the catalog, select cards, and record what it would dispatch.
   - Downstream prompts are unchanged in this dry-run mode.
   - This tests selection quality with almost no behavioral risk.
3. **Planner-dispatched Critic cards**
   - Low risk because Critic is read-only.
   - Useful once prompt-derived evidence, causality, and source-routing checks are available.
   - Helps catch invalid shortcut successes before changing generation behavior.
4. **Planner-dispatched Opt cards**
   - Natural fit because Opt already reasons about variables, objectives, bounds, and evidence.
   - Seed cards can immediately improve variable selection and parameter-effectiveness checks.
5. **Planner-dispatched writer cards**
   - Higher leverage but also higher risk because they shape source generation.
   - Start with narrow action/body cards, then expand to scene/rendering.

This order allows Planner to learn which cards are useful as evaluative and diagnostic context before relying on them to
shape generated source.

### Suite Setup

Build or refresh the debug-card catalog once per suite, similar to the Genesis context pack. The catalog should be
available to Planner and recorded in each case workspace, but it should not be exposed as a free-form file tree for every
downstream subagent.

Inputs:

- card YAML files
- suite physics mode
- effective deformable/IPC config

Outputs:

- `context/simdebug/catalog.json`
- card references under each case workspace
- optional cached selected-card summaries for Planner audit

### Planner

Planner owns all card selection and all card dispatch decisions. It should read the catalog, select all cards it judges
relevant to the current global case state, and decide which cards to distribute for the next action.

Planner use cases:

- Include known physical structure patterns in `planner_output`.
- Include likely failure modes and required evidence.
- Route failures to the right owner.
- Decide whether Opt is appropriate.
- Decide which cards should be sent to each writer, Critic, or Opt call.
- Revise dispatched cards after new execution, visual, Critic, or Opt evidence arrives.
- Stop dispatching cards that prove irrelevant or harmful in the current episode.

Planner should record each dispatch decision in a case-level audit report:

```json
{
  "turn": 4,
  "target": "opt",
  "selected_cards": [
    {
      "id": "fem_dual_plunger_leveling",
      "reason": "prompt tags: fem_ipc, precision_control, dual_actuator"
    },
    {
      "id": "opt_param_effectiveness",
      "reason": "Opt action requested; generated action code contains clamps"
    }
  ]
}
```

### Writers

Writers should not retrieve cards directly. Each writer receives only the owner-relevant card bundle that Planner
includes in the worker prompt or repair brief.

Examples:

- `body`: material ranges, collision geometry, initial clearance, mesh/XML suitability.
- `action`: physical control paths, actuator/DOF APIs, timing schedules, forbidden state writes.
- `scene`: solver/contact settings and IPC/FEM setup constraints.
- `rendering`: evidence clarity requirements only, not Opt variables.

### Critic

Critic should not retrieve cards directly. Planner should dispatch cards whose guidance, restriction, and check sections
clarify task-specific evidence requirements.

Critic use cases:

- Reject invalid physical shortcuts.
- Check that metrics reflect observed behavior.
- Assign source-aware repair ownership.
- Identify when numeric success is not visually or physically credible.

### Opt

Opt should not retrieve cards directly. Planner should dispatch Opt cards when invoking the Opt subagent, and may update
the dispatched bundle after baseline execution, Critic feedback, or an earlier Opt report.

Opt use cases:

- Select compact variable sets.
- Choose bounded ranges and log scales.
- Define shaped objective terms.
- Check parameter effectiveness.
- Decide when a case is structural rather than parameter-limited.

## Executable Probes

Some cards should point to executable probes. These probes turn human experience into evidence rather than
only text.

High-value first probes:

1. **Forbidden state-write scanner**
   - Detect post-initialization direct pose/qpos/qvel writes on passive task objects.
2. **Built-in asset scanner**
   - Detect prohibited Genesis built-in asset paths and texture dependencies.
3. **Opt parameter effectiveness smoke test**
   - Perturb one or two active variables and confirm metrics report changed effective values.
4. **Metrics coverage check**
   - Confirm the case records shaped task metrics, not only a binary success flag.
5. **IPC/FEM initial geometry sanity check**
   - Check obvious bbox clearances, initial overlaps, and asset scale mismatches before long rollouts.

Probe results should be written as small reports and fed back into Planner's global case state:

```text
reports/simdebug_probe_report.json
```

## Card Authoring And Migration Workflow

The first card set should mostly come from prompt migration. Human-authored and case-distilled cards can then fill gaps
that the prompts do not cover.

Initial migration workflow:

1. Inventory long prompt constants and policy sections in `code_agent/prompts/`.
2. Split each section into atomic rules.
3. Classify each atom as `keep_in_prompt`, `simdebug_card`, or `api_context`.
4. Create production YAML cards under the appropriate `cards/<primary_agent>/` category, preserving prompt provenance.
5. Add cards to `catalog.json` and audit Planner candidate selection before removing the corresponding static prompt
   text.
6. Replace prompt-derived static text with compact general hooks such as "follow Planner-dispatched human debugging
   cards".
7. Compare prompt sizes and representative suite behavior before enabling live dispatch.

Human-authored workflow:

1. Maintainer writes or edits a card from a known failure, successful repair, or advisor suggestion.
2. Card includes provenance, confidence, and at least one validation target when available.
3. A lightweight validator checks schema and required fields.
4. Card is added to `catalog.json`.
5. Planner can select and dispatch it in future suite runs.

Later case-distillation workflow:

1. After a case finishes, a summarizer proposes candidate cards from:
   - critic report
   - Opt report
   - execution logs
   - repair history
   - final metrics/video evidence
2. Human approves, edits, or rejects the proposed card.
3. Approved cards enter the library with provenance.

## Evaluation Plan

Evaluate this as a Planner-orchestrated guidance layer, not as a model-training project.

Suggested ablations:

1. Baseline current pipeline.
2. Static prompt additions only.
3. Planner-selected unified cards.
4. Planner-selected unified cards plus selected probes.

Metrics:

- final pass rate under fixed repair budget
- first-pass execution success
- average repair rounds
- Opt useful-result rate
- structural-failure vs parameter-failure routing accuracy
- number of invalid shortcut acceptances caught by Critic
- GPU time per passing case
- human review time per accepted card
- dispatch precision: how often Planner-selected cards were judged relevant after the episode
- prompt size reduction by role after migration
- prompt-regression rate: failures introduced by removing guidance from static prompts

Early target suites:

- deformable precision column/block-like cases
- rigid/articulated contact cases with XML actuators
- IPC rigid-soft coupling cases

## Milestones

### Milestone 1: Prompt Audit And Classification

- Inventory `code_agent/prompts/common.py`, `ipc.py`, `planner.py`, `worker.py`, `critic.py`, and `opt.py`.
- Mark each long guide/policy clause as `keep_in_prompt`, `simdebug_card`, or `api_context`.
- Identify prompt clauses that are duplicated across roles and should become shared cards.
- Produce a migration report before changing prompt behavior.

### Milestone 2: Card Library Skeleton And Full Prompt Migration

- Add `code_agent/context/simdebug/cards/` with primary-agent subdirectories.
- Define YAML schema and catalog builder.
- Convert every prompt clause classified as `simdebug_card` into a production card.
- Preserve prompt provenance for every prompt-derived card.
- Add optional human-authored cards for gaps not covered by existing prompts.

### Milestone 3: Planner Selection And Dispatch Dry Run

- Add deterministic Planner-side card selection.
- Record role-specific dispatch bundles for Critic, Opt, and owner-specific writer prompts without changing downstream
  prompts at first.
- Record selected cards, target roles, and dispatch reasons in each case report for auditability.
- Keep downstream agents file-decoupled from the card library.

### Milestone 4: Prompt Slimming And Live Dispatch

- Replace prompt-derived prompt sections with compact general hooks.
- Enable Planner-dispatched Critic cards first.
- Enable Planner-dispatched Opt cards next.
- Enable Planner-dispatched writer cards after Critic/Opt dispatch quality is acceptable.
- Keep rollback flags for both prompt migration and card dispatch.

### Milestone 5: First Probes

- Implement built-in asset scanner.
- Implement forbidden state-write scanner.
- Implement Opt parameter effectiveness check.
- Feed probe summaries into Planner, then let Planner decide which evidence and cards to dispatch to Critic and Opt.

### Milestone 6: Evaluation

- Run ablations on a fixed subset of deformable precision cases.
- Compare pass rate, repair rounds, Opt status distribution, and invalid-shortcut rate.
- Review which cards Planner selected, which cards it dispatched, and whether they helped or hurt.
- Review whether prompt-derived cards preserved the useful behavior of the original prompt clauses.

## Risks And Mitigations

### Risk: Cards Become Another Long Prompt

Mitigation: do not cap Planner's selected relevant cards by an arbitrary number, but require Planner to summarize,
deduplicate, group, and role-filter cards before dispatch. Full card text remains available to Planner by path, while
downstream agents receive synthesized role-specific bundles rather than the entire library.

### Risk: Bad Cards Pollute The System

Mitigation: require provenance, confidence, and validated cases. Prefer high-confidence cards and keep experimental
cards disabled by default.

### Risk: Restrictions Over-Constrain Generation

Mitigation: distinguish hard restrictions from soft guidelines. Restrictions should target invalid shortcuts and
evidence requirements, not arbitrary stylistic preferences.

### Risk: Planner Dispatch Adds Irrelevant Advice

Mitigation: hard-filter by physics mode and target role before text scoring. Log selected cards, dispatch reasons, and
target roles, then evaluate dispatch quality during ablations.

### Risk: Prompt Migration Removes Useful Always-On Context

Mitigation: migrate prompts gradually. Keep short universal anchors in prompts, run dry dispatch before live dispatch,
and keep rollback flags so prompt-derived static prompt sections can be restored if pass rate drops.

### Risk: Prompt-Derived Cards Preserve Old Prompt Bloat In A New Format

Mitigation: split prompt text into atomic cards and force each card to declare scopes, physics modes, task tags,
failure tags, and a compact summary. Do not migrate a whole long prompt section as one giant card.

### Risk: Probes Increase Runtime Cost

Mitigation: start with static or low-cost probes. Gate expensive FEM/IPC probes behind failure evidence or explicit
debug modes.

## Recommended Initial Card Set

Initial cards should first come from full prompt migration, then from failures already common in the current pipeline.
All prompt clauses classified as cards should become production cards; the lists below are examples of important prompt-derived
families rather than a fixed quota:

Prompt-derived cards with guidance-heavy sections:

- FEM material selection from `FEM_MATERIAL_SELECTION_GUIDE`.
- Rigid IPC coupling-mode selection from `RIGID_IPC_COUPLING_GUIDE`.
- External-articulation MJCF setup from `EXTERNAL_ARTICULATION_MJCF_GUIDE`.
- IPC failure diagnosis from `IPC_FAILURE_DIAGNOSTIC_GUIDE`.
- Source-aware repair routing from `SOURCE_AWARE_REPAIR_GUIDE`.
- Opt variable range and log-scale selection from `opt.py`.
- Shaped metric design and evidence selection from `opt.py` and `critic.py`.

Prompt-derived cards with restriction/check-heavy sections:

- Physical causality restrictions from `PHYSICAL_CAUSALITY_CONTRACT`.
- Collision/contact restrictions from `COLLISION_CONTACT_CONTRACT`.
- Scale and asset-use restrictions from `SCALE_POLICY_GUIDE` and `BUILTIN_ASSET_POLICY_GUIDE`.
- Rendering evidence restrictions from `RENDER_CLARITY_GUIDE`.
- No camera/rendering optimization inside Opt from `opt.py`.
- No active Opt variable that is ignored, clamped to a constant, or missing from metrics.
- No numeric success without visual evidence from `critic.py` and `opt.py`.

Human-authored gap-fill cards:

- FEM dual-plunger leveling variable and metric selection.
- Precision-control binary-metric insufficiency.
- Contact-driven passive-object motion diagnosis.
- Suite-specific deformable column/block tuning patterns.

## Expected Outcome

This plan should give `code_agent` a more effective path for using human experience:

- API guides teach agents how to call Genesis.
- Guideline cards teach agents how experienced developers diagnose and tune simulations.
- Restriction cards teach agents how to reject invalid shortcuts and incomplete evidence.
- Probes convert important restrictions into executable checks.
- Planner decides which card guidance each downstream agent should see at each point in the episode.
- Static prompts become shorter and more general, while prompt-derived knowledge remains available through auditable
  Planner dispatch.

The result is a human-experience layer that is more precise than static prompt engineering and cheaper than immediate
model fine-tuning.
