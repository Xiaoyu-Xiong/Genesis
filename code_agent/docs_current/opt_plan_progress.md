# Opt Subagent Plan And Progress

This document tracks the design and implementation progress for adding a dedicated Opt Codex subagent to `code_agent`.
The current direction is subagent-first: `code_agent/opt/` should provide optimization protocols, schemas, runners, and
backend utilities, while the Opt subagent decides how to use them for a specific generated case.

## Updated Direction

The previous framing treated optimization mostly as a fixed post-generation runner:

```text
generate case -> expose parameters -> run CMA-ES -> render best
```

That is useful as an implementation scaffold, but it is too rigid for a modern agentic pipeline. The revised direction is:

```text
Planner decides whether optimization is needed.
Planner invokes an Opt Codex subagent with task constraints and budget.
Opt inspects generated code and artifacts.
Opt parameterizes, instruments, optimizes, diagnoses, and reports.
Planner decides accept, continue optimization, or route a rewrite.
```

The key change is that optimization strategy should be chosen by the Opt subagent from evidence, not hard-coded as a
global pipeline policy.

## Motivation

The current Planner-led pipeline can generate runnable Genesis scenes, but complex physical tasks still depend on
sensitive values in generated modules. Examples include:

- Grasp target positions.
- Release timing and placement.
- PD gains, damping, motor force limits, and velocity limits.
- Initial pose, lean angle, spacing, preload, mass distribution, and other balance-sensitive setup parameters.
- Material stiffness, density, friction, restitution, and damping.
- Existing XML/MJCF actuator, joint, and geom scalar attributes such as `kp`, `forcerange`, `damping`, `armature`,
  `range`, and friction.
- Solver/contact parameters such as substeps, contact distance, and tolerance.

Planner should not need to guess these values. The original generation agents also should not be forced to expose every
possible variable in advance. Instead, Planner can call Opt when a case appears parameter-sensitive. Opt then reads the
generated code and decides which values should become optimization variables.

## Roles

### Planner

Planner owns orchestration and final routing.

Planner should:

- Decide whether a case needs Opt.
- Provide the original prompt, task intent, constraints, allowed edits, forbidden edits, and rollout budget.
- Accept or reject Opt's result.
- Decide whether to ask Opt for another pass.
- Route structural failures back to scene/body/action/rendering agents.

Planner should not:

- Pick individual numeric parameters such as `kp`, friction, or release offsets.
- Hard-code staged optimization plans.
- Treat Opt success as final acceptance without critic/evidence review.

### Opt Codex Subagent

Opt owns the optimization pass.

Opt should:

- Inspect generated source, contracts, artifacts, reports, metrics, and videos.
- Identify optimizable physical/control/setup parameters and their owners in `scene`, `body`, `action`, or validated XML
  scalar patches.
- Patch generated code to read opt params when needed.
- Create or revise `target_spec.json`, `opt_space.json`, and `default_opt_params.json`.
- Choose an optimizer backend and budget use.
- Run baseline, trial, and best render rollouts as needed.
- Diagnose whether failure is parameter-level, metric-level, objective-level, or structural.
- Return structured evidence and recommendations to Planner.

Opt should not:

- Change task semantics.
- Add hidden constraints, attachments, suction, or direct post-initialization object state writes.
- Edit `src/rendering.py` or optimize rendering/camera/visual-only variables.
- Replace assets or mechanisms without reporting `needs_rewrite`.
- Become a second Planner.

### Critic

Critic remains responsible for checking physical faithfulness and visual/source evidence. It should use Opt reports as
evidence, not as a substitute for judgment.

## Planner To Opt Handoff

Planner should send a compact structured request. Required fields:

- `case_dir`
- `original_prompt`
- `planner_intent`
- `allowed_edits`
- `forbidden_changes`
- `optimization_budget`
- `success_criteria`

Recommended fields:

- `suspected_failure_modes`
- `priority_parameters`
- `must_render_baseline`
- `must_render_best`
- `max_wall_time_sec`

Example:

```json
{
  "case_dir": "code_agent/workspaces/.../fetch_robot_rigid_grasp",
  "original_prompt": "Create a Fetch-style manipulator grasping and releasing a ball into a dish.",
  "planner_intent": "Optimize if generated behavior is limited by bounded continuous parameters.",
  "allowed_edits": [
    "src/action.py",
    "src/body.py for material/contact/initial-setting hooks",
    "src/scene.py for solver/contact/timestep hooks",
    "assets/xml/**/*.xml for validated scalar actuator/joint/geom patches",
    "contracts/*.json",
    "reports/*.json",
    "artifacts/opt_*"
  ],
  "forbidden_changes": [
    "Do not change task semantics.",
    "Do not directly write dynamic object state after initialization.",
    "Do not add hidden constraints or attachments.",
    "Do not edit src/rendering.py or optimize rendering/camera/visual-only variables.",
    "Do not change XML topology during Opt."
  ],
  "optimization_budget": {
    "max_rollouts": 20,
    "backend": "gpu",
    "render_baseline": true,
    "render_best": true
  },
  "success_criteria": [
    "Object is manipulated through physical contact.",
    "Target behavior is visible in metrics and video.",
    "Original prompt constraints are preserved."
  ]
}
```

## Opt To Planner Report

Opt should return:

- `status`
- `edited_files`
- `optimized_variables`
- `baseline`
- `best`
- `diagnosis`
- `evidence`
- `recommendation`

Allowed statuses:

- `success`
- `partial_success`
- `needs_more_optimization`
- `needs_rewrite`
- `failed`

Example success:

```json
{
  "status": "success",
  "edited_files": [
    "src/action.py",
    "contracts/target_spec.json",
    "contracts/opt_space.json"
  ],
  "optimized_variables": [
    "target.grasp_z_offset_m",
    "target.release_y_offset_m",
    "gripper.closed_command_m"
  ],
  "baseline": {
    "success": false,
    "score": -0.48,
    "metrics_path": "artifacts/opt_baseline/metrics.json",
    "video_path": "artifacts/opt_baseline/render.mp4",
    "params_path": "contracts/default_opt_params.json",
    "summary": "Baseline misses the target gates."
  },
  "best": {
    "success": true,
    "score": 5.85,
    "params_path": "contracts/best_opt_params.json",
    "metrics_path": "artifacts/opt_best/metrics.json",
    "video_path": "artifacts/opt_best/render.mp4",
    "summary": "Optimized rollout satisfies the task."
  },
  "diagnosis": "The case was parameter-limited and is now successful.",
  "recommendation": "Planner can proceed to critic/final acceptance."
}
```

`baseline` and `best` now use fixed fields rather than open-ended dictionaries, because Codex structured output rejects
schemas whose `required` and `properties` are not strictly aligned.

Example rewrite request:

```json
{
  "status": "needs_rewrite",
  "likely_owner": "body",
  "diagnosis": "The gripper geometry cannot enclose the ball; all trials fail before contact.",
  "evidence": [
    "Pad-to-ball distance never reaches tolerance.",
    "Changing controller targets does not improve contact.",
    "Rendered baseline shows one-sided gripper geometry."
  ],
  "recommendation": "Ask body/action agents to regenerate gripper geometry and control handles."
}
```

## What Lives In `code_agent/opt/`

`code_agent/opt/` should not encode one rigid optimization policy. It should provide reusable capabilities:

- `cli.py`: temporary command-line facade for early debugging. It parses user/manual flags, constructs
  `OptAgentRequest`, calls `run_opt_agent`, and prints the result. It can be removed once Planner integration is stable.
- `agent.py`: Planner-facing Python harness. It builds the Opt prompt, invokes the Codex Opt subagent, parses the
  structured result, and records `reports/opt_subagent_report.json`.
- `code_agent/prompts/opt.py`: Opt subagent role prompt, safety rules, available runner commands, and final report
  instructions.
- `types.py`: request/result dataclasses.
- `contracts.py` and `objective.py`: contract validation, vector/payload conversion, and metric scoring.
- `runner.py`: low-level numerical optimization loop and budget/runtime option resolution.
- `trials.py`: single-rollout execution, current-parameter writing, metric loading, trial scoring, and best rendering.
- `reports.py`: trace, `opt_report.json`, and `verification_report.json` writing.
- `optimizers/cma_es.py`: CMA-ES backend. Future optimizer backends should be sibling modules under `optimizers/`.

`code_agent.opt.agent.run_opt_agent` is the Planner-facing API that invokes Codex. `code_agent.opt.cli` is a temporary
manual wrapper around that API. The `code_agent.cli run-opt` command is a lower-level numerical runner that the Opt
subagent may call after it has prepared contracts and parameter hooks.

## Parameter Scope

Opt should be free to propose initial-setting, layout, geometric, schedule, control, actuator, material, contact, XML
scalar, or solver parameters owned by `scene.py`, `body.py`, `action.py`, or existing XML/MJCF assets, as long as they
are bounded, physically meaningful, and connected to observable metrics. `rendering.py` is excluded from the
optimization surface; camera or visual-evidence failures should be routed back to Planner as rendering repair needs.

Examples:

- `target.grasp_z_offset_m`
- `target.release_x_offset_m`
- `control.gripper_kp`
- `control.arm_kv`
- `control.motor_force_limit`
- `initial.card_lean_angle`
- `layout.stack_gap`
- `material.ball_friction`
- `material.soft_body_density`
- `material.youngs_modulus`
- `xml.actuator.wrist_kp`
- `xml.joint.hinge_damping`
- `solver.substeps`
- `contact.d_hat`

Mechanical parameters are not second-class. They are often the most important variables for inverse design, especially
for deformables, contact-rich manipulation, and articulated control.

## Current Implementation Status

Implemented:

- `code_agent/opt/` generic runner, contract utilities, objective evaluator, and CMA-ES backend.
- `code_agent/opt/optimizers/cma_es.py` optimizer backend.
- `CONFIGS.opt` for shared Opt-agent request defaults, runner timeout/render/path defaults, fallback normalized sigma,
  and CMA-ES population-size rule constants.
- `code_agent.opt.cli` temporary manual CLI and `code_agent.opt.agent.run_opt_agent` Codex invocation wrapper.
- `code_agent.prompts.opt` role prompt for the autonomous Opt Codex subagent.
- `code_agent.cli run-opt` CLI entry point.
- Formal schemas under `code_agent/specs/opt_schema/`.
- `opt_subagent_report.schema.json` for the Codex Opt subagent's final structured output.
- `target_spec.json`, `opt_space.json`, `opt_params.json`, trace, report, and verification formats.
- Baseline trial support.
- Isolated trial artifacts under `artifacts/opt_trials/`.
- `best_opt_params.json`, `opt_report.json`, and `verification_report.json`.
- Adaptive default population size.
- Per-variable normalized `initial_sigma`.

Previously hand-coded generated-case adapters were removed. The current path relies on Codex to inspect and adapt each
workspace rather than matching the two earlier test cases.

Latest progress:

- Replaced the single-backend `cma_es/optimizer.py` layout with `optimizers/cma_es.py`, so future optimizers can be added
  as sibling modules.
- Removed the separate `workspace.py` helper and then removed all generated-case adapters/templates/source-patching
  recipes. Python no longer hard-codes the two test cases.
- Replaced adapter dispatch with a Codex Opt subagent invocation using `CodexExecRequest`, `opt_model`, `opt_sandbox`,
  and `opt_timeout_sec`.
- Split the previous 500+ line `runner.py` into three coarse-grained files: `runner.py` for the numerical optimization
  loop, `trials.py` for single rollout execution, and `reports.py` for trace/report artifacts.
- Moved scattered Opt defaults into `CONFIGS.opt`, including request defaults, runner artifact paths, fallback
  `initial_sigma`, and CMA-ES default population-size constants.
- Verified imports, module help, and py-compile for the Opt package after the split.

## Outdated Assumptions Removed

The following assumptions should no longer guide future design:

- The original generation agents must always emit perfect opt contracts before Opt can be useful.
- Planner should hard-code staged optimization, restart profiles, or variable-freezing policies.
- Opt is just a single `run-opt` call.
- CMA-ES is the permanent optimizer identity of the project.
- Optimization variables are mostly geometric or schedule variables.

These may still happen in a specific case, but they should be choices made by Opt or Planner from evidence.

## Next Milestones

### M1: Opt Subagent Prompt And Invocation Contract

Status: implemented.

- Write the Opt subagent role prompt.
- Define the exact Planner-to-Opt request format. The prototype is `OptAgentRequest`.
- Define the Opt-to-Planner report format. The prototype is `OptAgentResult` plus `reports/opt_subagent_report.json`.
- Specify allowed edit scopes and forbidden changes. Initial defaults live in `types.py`.
- Include examples of `success`, `partial_success`, `needs_more_optimization`, `needs_rewrite`, and `failed`.

Gate:

- Planner can invoke Opt with a structured brief without embedding optimization strategy.

### M2: Workspace Adaptation Playbook

Status: partial.

- Document how Opt should inspect generated code.
- Document safe parameter-hook patterns for `scene.py`, `body.py`, `action.py`, and constrained XML scalar patches.
- Document how to expose controller gains, material parameters, contact parameters, and solver parameters.
- Document how to add metrics without changing task semantics.

Current code prompts Codex to inspect and adapt arbitrary generated workspaces. More detailed examples of safe hooks for
body/material and scene/solver variables still need to be documented.

Gate:

- Opt can adapt a generated case that had no pre-existing opt contracts.

### M3: Agentic Optimization Pass

Status: implemented and smoke-tested on two generated rigid/control cases.

- Run Opt as a Codex subagent on an existing generated case.
- Let Opt choose variables and bounds from code inspection.
- Let Opt decide whether to run `run-opt`, manual baseline renders, additional trial batches, or return `needs_rewrite`.
- Require an evidence-rich result for Planner.

Gate:

- Opt improves or correctly diagnoses a case without the developer manually choosing variables.

Smoke test on `cases_active_rigid_20260515_173129`:

- `allegro_hand_soft_ball`: Codex Opt classified the case as parameter-limited, used the existing five-variable
  action-space contract, ran CMA-ES, rendered baseline and best videos, and improved from failing baseline
  `score=3.2263952` to verified best `score=3.35`, `success=true`. Evidence lives in
  `artifacts/opt_agent_baseline/`, `artifacts/opt_agent_best_full/`, and `reports/opt_subagent_report.json`.
- `fetch_robot_rigid_grasp`: Codex Opt repaired the case-level opt interface, exposed six action variables, ran CMA-ES,
  rendered baseline and best videos, and improved from failing baseline `score=-0.44128689841397284` to verified best
  `score=5.85`, `success=true`. Evidence lives in `artifacts/opt_agent_baseline/`,
  `artifacts/opt_agent_best_full/`, and `reports/opt_subagent_report.json`.
- During testing, `opt_subagent_report.schema.json` needed one self-correction: the first open-ended
  `baseline`/`best` schema was rejected by Codex structured output. It is now a strict fixed-field evidence object.

Current gap: more fresh generated cases are still needed to measure whether Codex reliably chooses good variables,
patches code safely, and diagnoses structural failures without pre-existing opt contracts.

### M4: Planner Integration

Status: implemented at action level; needs end-to-end suite validation.

- Added Planner action `run_opt`, routed through `RuntimeActionHandler.run_opt`.
- `run_opt` is gated by the suite's effective Opt setting. It defaults to `CONFIGS.opt.enabled` and can be overridden
  with `run-suite --enable-opt` or `run-suite --disable-opt`; when disabled, Planner must use the normal
  execution/critic/repair path.
- Planner prompt now describes when to use Opt and when to avoid it.
- The episode state now tracks Opt enabled/status/attempts/latest result/history.
- On `success`, `partial_success`, or `needs_more_optimization`, the handler syncs `best_opt_params.json` to
  `current_opt_params.json` and marks the case for `run_execution` so root artifacts are regenerated from the selected
  optimized parameters before Critic acceptance.
- Critic prompts now include Opt reports and opt parameter payloads when present, but still judge the current execution
  artifacts independently.
- `run_opt_agent` now recovers fresh lower-level `reports/opt_report.json` evidence when the Codex Opt subagent times
  out or exits nonzero after running optimization. This prevents completed CMA-ES traces and best-parameter payloads
  from being discarded just because the reporting subagent missed its final JSON deadline.
- Planner/Opt decision guidance is now more general: Planner is prompted to call Opt for runnable cases with continuous
  measurable residuals over action, initial setting/layout, material/contact, actuator/XML scalar, or solver/contact
  parameters, and to route structural failures to repair/regeneration.
- Opt contracts now allow `owner: "xml"` plus `actuator`, `initial`, `layout`, and `geometry` groups. XML edits are
  restricted to validated scalar patches on existing actuator, joint, or geom attributes; topology changes should return
  `needs_rewrite`.
- Opt success now requires visual evidence when best rendering is requested. The Opt prompt requires a
  `video_checked=...` evidence item, and the Planner-facing harness downgrades `success`/`partial_success` to
  `needs_more_optimization` if the best video path or explicit video/frame inspection evidence is missing.
- The low-level CMA-ES runner now supports agent-selected `strategy.phases`, `strategy.restarts`, and
  `strategy.early_stop` entries in `contracts/opt_space.json`. Reports include strategy diagnostics, sparse-objective
  warnings, and best-parameter boundary warnings.
- The CMA-ES backend now uses `pycma` while preserving the existing ask/tell wrapper. Runner scoring uses
  direction-aware worst scores for invalid trials, rejects `transform: "custom"`, treats missing `success_criteria` as
  non-successful verification, and repeats the selected best params before final rendering with strict-majority success
  for better noise robustness.
- The old pipeline is preserved: if `CONFIGS.opt.enabled` is false or Planner never chooses `run_opt`, generation,
  execution, critic, and repair proceed as before.

Gate:

- One suite case runs generation -> optional Opt -> critic with no manual intervention.

### M5: Broader Evaluation

Status: pending.

- Compare no-Opt generation against Opt-enhanced generation.
- Track success rate, number of rollouts, score improvement, and diagnosis quality.
- Include deformable material tuning, articulated control tuning, and coupled rigid/deformable cases.

Gate:

- The project can report optimization benefit and failure routing quality across a small benchmark.

## Design Risks

- Opt may over-edit generated code and blur the boundary between optimization and generation.
- Numeric success may disagree with visual/physical evidence.
- Bad objectives may reward unrealistic parameter values.
- High-dimensional search spaces may waste rollout budget.
- Structural failures may be misdiagnosed as parameter failures.

Mitigations:

- Keep Planner as final orchestrator.
- Require Opt to report edited files and physical constraints.
- Keep Critic independent.
- Prefer small, meaningful parameter sets unless Opt has evidence for broader search.
- Route `needs_rewrite` when parameter optimization is not the right tool.
