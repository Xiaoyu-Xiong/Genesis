# Opt Codex Subagent

`code_agent/opt/` is the toolbox and protocol layer for a dedicated Opt Codex subagent. The subagent is responsible for
optimization-related work on generated Genesis cases when, and only when, Planner decides that parameter optimization is
worth invoking.

The important architectural choice is that optimization strategy should live in the Opt subagent's reasoning, not in a
hard-coded pipeline. The repository should provide reliable tools, schemas, runners, reports, and examples. The Opt
subagent decides which parameters to expose, which optimizer to use, how many trials to run, whether to revise the search
space, and whether the case should be sent back to Planner for structural repair.

## Core Principle

The normal generation pipeline should remain valid:

```text
Planner -> scene/body/action/rendering agents -> execution/critic
```

Optimization is an optional branch:

```text
Planner
  -> generated case
  -> optional Opt Codex subagent
  -> Planner accepts, asks for more optimization, or routes a rewrite
```

This means simple cases do not pay an optimization cost, while contact-rich or parameter-sensitive cases can receive a
specialized optimization pass.

## What Opt Is

The Opt subagent is:

- A Codex subagent specialized for optimization, parameterization, trial execution, and diagnosis.
- A reader and editor of generated case workspaces.
- A user of optimizer backends such as CMA-ES.
- A producer of evidence: baseline metrics, trial traces, best parameters, rendered best videos, and diagnosis reports.
- A source of recommendations back to Planner.

The Opt subagent is not:

- A replacement for Planner.
- A replacement for the original scene/body/action/rendering agents.
- A hidden task rewriter.
- A mechanism for adding non-physical shortcuts, hidden attachments, or post-initialization state writes.
- A final judge of task acceptance. Planner and Critic still own final acceptance.

## Planner Responsibilities

Planner decides whether to call Opt and how to use the result.

Planner should call Opt when:

- The generated case is runnable: code executes, required entities exist, video/metrics are produced, and the physical
  causal story is basically plausible.
- The failure looks like a continuous parameter residual rather than a missing structure: distance is close but off,
  speed/angle/timing is wrong, contact happens in the wrong place, a PD gain is unstable, an initial balance setting is
  slightly wrong, or friction/damping/density is sensitive.
- There are real bounded continuous parameters that Opt can safely expose. These may live in action schedules,
  body/material setup, initial placement/layout, scene/contact settings, actuator gains/limits, or validated XML/MJCF
  scalar attributes.
- The prompt explicitly asks for inverse design, controllability, tuning, or target matching.
- A critic or execution report suggests that repeated local repair is converging to "almost works, but wrong numbers".

Planner may skip Opt when:

- The default generated rollout already satisfies the task.
- The failure is clearly structural, such as missing required bodies, invalid assets, missing physical affordances or
  continuous parameter hooks, wrong joint axes, caged/stuck task objects, impossible geometry, absent metrics, or
  invisible target behavior.
- The task is purely visual or rendering-only.
- The optimization budget is too small to produce useful evidence.

Planner should provide Opt with:

- The case directory.
- The original prompt and Planner intent.
- Success criteria and non-negotiable physical constraints.
- Allowed and forbidden edits.
- Optimization budget and backend preference.
- Any specific suspicion, such as "release placement is failing" or "material stiffness looks wrong".

Planner should decide after Opt returns:

- Accept the optimized best parameters and continue to final critic.
- Ask Opt for another optimization pass with a revised budget or narrower goal.
- Ask the original agents to rewrite code or assets based on Opt's diagnosis.
- Mark the case inconclusive or failed if optimization cannot repair it safely.

## Opt Subagent Responsibilities

The Opt subagent owns the optimization pass end to end.

It should:

- Inspect generated `src/scene.py`, `src/body.py`, `src/action.py`, `src/main.py`, existing contracts, artifacts,
  reports, and metrics. It may read `src/rendering.py` only to understand available evidence, not as an edit target.
- Identify sensitive values that are safe to optimize.
- Decide whether parameters belong to `scene`, `body`, or `action`.
- Decide whether actuator or joint tuning should be applied through runtime Genesis APIs or through a constrained XML
  scalar patch.
- Patch generated code so it can read opt parameters while remaining runnable without Opt.
- Emit or revise `contracts/target_spec.json`, `contracts/opt_space.json`, and
  `contracts/default_opt_params.json`.
- Choose an optimizer backend. Version 1 has CMA-ES available; future versions may add random search, Bayesian
  optimization, trajectory optimization, differentiable simulation, or RL.
- Run baseline and candidate rollouts, usually without render.
- Render baseline and best results when requested or when visual evidence is needed.
- Inspect the best render/video before reporting `success` or `partial_success`. Numeric success is not sufficient; the
  final evidence must include a `video_checked=...` item describing the video or sampled frames and the visual outcome.
- Write trace and report artifacts.
- Diagnose whether remaining failure is parameter-level, objective-level, metric-level, or structural.
- Return a concise structured result to Planner.

It may patch generated modules, but it must keep edits local and explain them. It should not silently transform a task
into a different task.

## Current Code Layout

`code_agent/opt/` now separates the human/manual CLI entry point, the Planner-facing agent harness, and reusable
optimization mechanics:

- `cli.py`: temporary command-line facade for early debugging. It parses flags such as `--case-dir`, `--max-rollouts`,
  and `--render-best`, constructs an `OptAgentRequest`, calls `run_opt_agent`, and prints the resulting JSON. It should
  not contain prompt text, Codex execution logic, report parsing, or optimization policy.
- `agent.py`: programmatic Planner-facing harness. It builds the Opt prompt, invokes the real Codex Opt subagent,
  parses the final JSON, normalizes it into `OptAgentResult`, and writes `reports/opt_subagent_report.json`. It should
  not parse CLI arguments or hard-code case-specific parameter choices.
- `code_agent/prompts/opt.py`: the Opt subagent role prompt, including inspection duties, safety constraints, allowed
  edits, runner commands, and final report requirements.
- `types.py`: Planner-to-Opt request and Opt-to-Planner result dataclasses.
- `contracts.py` and `objective.py`: opt contract validation, vector/payload conversion, and metric scoring.
- `runner.py`: low-level numerical optimization orchestrator. It loads contracts, resolves budget/runtime options, runs
  baselines, selects/renders the best result, and writes reports.
- `strategy.py`: parses generic strategy knobs from `contracts/opt_space.json`, including phases, restarts, and
  early-stop settings.
- `search.py`: executes the CMA-ES phase/restart search loop using the chosen strategy.
- `trials.py`: single-rollout execution helper. It writes current opt params, runs generated `src/main.py`, loads
  metrics, scores a trial, and optionally renders the best result.
- `reports.py`: trace, `opt_report.json`, and `verification_report.json` writing.
- `optimizers/cma_es.py`: the CMA-ES backend. Future optimizer backends should live under `optimizers/` as sibling
  modules.
- `code_agent/configs.py`: shared Opt defaults under `CONFIGS.opt`, including manual Opt-agent request defaults, runner
  timeout/render/path defaults, fallback normalized sigma, and CMA-ES population-size rule constants.

This layout intentionally keeps parameterization decisions out of Python hard-coded adapters. The Codex Opt subagent is
responsible for inspecting each generated workspace, deciding whether and how to expose variables, and calling the
generic runner when appropriate.

## Editing Boundaries

Opt may usually edit:

- `src/action.py` to expose timing, target, controller, PD, force, and schedule parameters.
- `src/body.py` to expose material, density, friction, restitution, geometry-scale, initial placement, and object
  parameter hooks.
- `src/scene.py` to expose carefully bounded solver/contact parameters when the value is a genuine simulation setting.
- `assets/xml/**/*.xml` only for validated scalar patches on existing actuator, joint, or geom attributes. Prefer
  runtime hooks such as `set_dofs_kp`, `set_dofs_kv`, and `set_dofs_force_range` when they can express the same tuning.
- `contracts/*.json` for optimization specs and parameter payloads.
- `reports/*.json` and optimization artifacts.

Opt must not edit:

- `src/rendering.py`, camera placement, lights, capture cadence, or visual-only parameters.

Opt should avoid editing:

- Generated assets, except for scalar XML/MJCF parameter patches that preserve the same bodies, joints, geoms,
  actuators, meshes, names, and joint axes. If the asset or mechanism is structurally wrong, Opt should report
  `needs_rewrite` to Planner.
- Task text, success semantics, or entity set.

If visual evidence is unclear because the camera or renderer is wrong, Opt should report that Planner needs a rendering
repair instead of turning rendering into an optimization variable.

Opt must not:

- Add hidden constraints, attachments, suction, teleportation, or direct dynamic-object state writes after
  initialization unless the original task explicitly asks for them.
- Remove required physical participants.
- Replace an articulated, deformable, or coupled simulation with a simpler fake.
- Overwrite unrelated user or worker changes.

## Planner To Opt Request

Planner should call the Opt subagent with a structured brief. A representative request is:

```json
{
  "case_dir": "code_agent/workspaces/.../fetch_robot_rigid_grasp",
  "original_prompt": "Create a Fetch-style manipulator grasping and releasing an orange rigid ball into a dish.",
  "planner_intent": "Optimize generated behavior if parameters, not structure, appear to limit success.",
  "allowed_edits": [
    "src/action.py",
    "src/body.py only for material/contact/initial-setting hooks",
    "src/scene.py only for solver/contact/timestep hooks",
    "assets/xml/**/*.xml only for validated scalar actuator/joint/geom patches",
    "contracts/*.json",
    "reports/*.json",
    "artifacts/opt_*"
  ],
  "forbidden_changes": [
    "Do not change task semantics.",
    "Do not directly write dynamic object state after initialization.",
    "Do not add hidden constraints or attachments.",
    "Do not edit src/rendering.py or optimize rendering/camera/visual-only variables.",
    "Do not change XML topology during Opt.",
    "Do not replace generated assets without reporting needs_rewrite."
  ],
  "optimization_budget": {
    "max_rollouts": 20,
    "backend": "gpu",
    "render_baseline": true,
    "render_best": true
  },
  "success_criteria": [
    "The object is manipulated by physical contact.",
    "The target behavior is visible in metrics and video.",
    "The optimized case preserves the original prompt constraints."
  ]
}
```

The request is guidance, not a rigid script. The Opt subagent can still decide how to parameterize, whether to run
baseline first, which optimizer to choose, and when to stop.

## Opt To Planner Result

Opt should return a structured summary:

```json
{
  "status": "success",
  "edited_files": [
    "src/action.py",
    "contracts/target_spec.json",
    "contracts/opt_space.json",
    "contracts/default_opt_params.json"
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
    "summary": "Baseline misses the lift/release gates."
  },
  "best": {
    "success": true,
    "score": 5.85,
    "params_path": "contracts/best_opt_params.json",
    "metrics_path": "artifacts/opt_best/metrics.json",
    "video_path": "artifacts/opt_best/render.mp4",
    "summary": "Optimized rollout completes the target behavior."
  },
  "diagnosis": "The case was parameter-limited. Grasp, lift, release, and final placement pass after tuning.",
  "recommendation": "Planner can proceed to critic/final acceptance."
}
```

Allowed statuses:

- `success`: optimized result satisfies metrics and available visual evidence.
- `partial_success`: optimization improved behavior but not enough for final acceptance.
- `needs_more_optimization`: current evidence suggests another opt pass is useful.
- `needs_rewrite`: parameter optimization is not the right fix; Planner should route a rewrite.
- `failed`: Opt could not run, contracts are invalid, rollouts fail, or evidence is inconclusive.

`baseline` and `best` are fixed-field evidence objects for Codex structured output compatibility. Use `null` for
unavailable fields. Put extra details in `summary` or `evidence` instead of adding ad hoc fields.

For `needs_rewrite`, Opt should include owner guidance through the existing schema fields:

```json
{
  "status": "needs_rewrite",
  "case_type": "rigid_grasp",
  "edited_files": [],
  "optimized_variables": [],
  "baseline": {
    "success": false,
    "score": -0.73,
    "metrics_path": "artifacts/opt_agent_baseline/metrics.json",
    "video_path": "artifacts/opt_agent_baseline/render.mp4",
    "params_path": "contracts/default_opt_params.json",
    "summary": "Baseline never reaches opposing contact."
  },
  "best": {
    "success": null,
    "score": null,
    "metrics_path": null,
    "video_path": null,
    "params_path": null,
    "summary": null
  },
  "diagnosis": "The gripper geometry cannot enclose the ball; likely owner is body/action rather than parameter tuning.",
  "evidence": [
    "Pad-to-ball distance never enters tolerance.",
    "Video shows one-sided approach with no opposing contact.",
    "Changing controller targets does not improve contact."
  ],
  "opt_report_path": null,
  "failures": ["structural_gripper_geometry"],
  "recommendation": "Ask body/action agents to regenerate gripper geometry and control handles."
}
```

## Parameterization Scope

Opt can expose any bounded scalar that is physically meaningful and read at the correct simulation lifecycle point. This
includes tasks with no obvious action policy, such as card-house, arch, stack, or balance scenes, when the remaining
failure is a fine initial-setting/material/contact parameter rather than a structural impossibility.

Common `action.py` variables:

- Phase timings and schedule fractions.
- End-effector, base, wrist, gripper, plate, or tool target positions.
- PD gains such as `kp`, `kv`, damping, motor gains, force limits, velocity limits, torque scales, impedance gains.
- External force or impulse magnitudes when the task allows them.

Common `body.py` variables:

- Object density, mass scale, friction, restitution, damping.
- FEM Young's modulus, Poisson ratio, material damping, bending/rod stiffness, plasticity parameters.
- Geometry scale, initial placement, lean angle, layout gap, preload, and center-of-mass offsets, if changing them
  preserves prompt semantics.
- Contact material parameters.

Common `scene.py` variables:

- Timestep, substeps, solver tolerances, contact stiffness, contact distance, IPC/contact settings.
- These should be bounded tightly and used only when they represent solver tuning rather than task cheating.

Common XML/MJCF scalar variables:

- Existing actuator `kp`, `ctrlrange`, and `forcerange` values.
- Existing joint `damping`, `armature`, and `range` values.
- Existing geom/contact scalar attributes such as friction, density, mass, `solref`, or `solimp` when meaningful.
- XML tuning should not add/remove/rename bodies, joints, geoms, actuators, meshes, defaults, or change joint axes.
  Prefer runtime Genesis control APIs when the same parameter can be applied after asset loading.

Use `scale: "log"` for positive variables spanning orders of magnitude, such as stiffness, damping, density, solver
tolerances, and many controller gains. Use `scale: "linear"` for signed offsets, schedule fractions, moderate friction
ranges, and target positions.

## Optimizer Backends

Version 1 exposes CMA-ES.

CMA-ES is useful because:

- It does not need gradients.
- It tolerates noisy metrics better than grid search.
- It fits low-dimensional continuous tuning.
- It can tune initial setting, layout, geometry, control, actuator, material, contact, XML scalar, and solver values
  through one contract format.

CMA-ES details currently supported:

- Variables are normalized to `[0, 1]`.
- The backend uses `pycma`'s bounded CMA-ES implementation instead of the earlier in-repo CMA-ES update code. Returned
  candidates are still validated inside `[0, 1]` before trial execution.
- `population_size` may be explicit or auto-selected. Explicit values must be at least 3 because pycma cannot update
  from smaller populations.
- If omitted, `population_size` uses:

```text
canonical = CONFIGS.opt.cma_es_population_base
          + floor(CONFIGS.opt.cma_es_population_log_multiplier * log(dim))
if dim <= CONFIGS.opt.cma_es_low_dim_threshold:
    population_size = clamp(
        canonical,
        CONFIGS.opt.cma_es_low_dim_min_population,
        CONFIGS.opt.cma_es_low_dim_max_population,
    )
else:
    population_size = canonical
```

- Each variable may define `initial_sigma` in normalized `[0, 1]` coordinates.
- Variables without `initial_sigma` use `CONFIGS.opt.runner_default_initial_sigma` (currently `0.25`).
- The backend initializes covariance with per-variable sigmas and then adapts normally.
- `strategy.phases` may optimize variable groups or named variables in stages. This lets Opt try action/control timing
  first, then material/contact, then scene/solver settings, without hard-coding that sequence into Planner.
- `strategy.restarts` may run multiple seeds or sigma scales. This is useful when one CMA-ES run can get stuck in a
  bad basin.
- `strategy.early_stop` may stop a phase/restart when success criteria are met or when several generations fail to
  improve by `min_delta`.
- The runner repeats the selected best parameter rollout before final rendering by default and uses the median repeated
  score plus strict-majority success for verification, reducing one-off noisy contact successes.
- The runner reports objective-shaping warnings when the objective is only binary, and boundary warnings when the
  selected best parameters cluster near search bounds.
- `transform: "custom"` is intentionally rejected. Generated code should write any custom scalar score into
  `metrics.json`, and `target_spec.json` should read that metric with `transform: "identity"`.
- Missing `success_criteria` means score can improve but verification success is false.
- Low-fidelity first is intentionally not part of the current runner strategy because changing duration, fps, or solver
  fidelity can make early evidence disagree with the final rendered rollout.

These settings are tools for Opt, not a fixed strategy. Opt may choose explicit values, run small probe batches, revise
the search space, or ask Planner for a rewrite.

## Case Contracts

Optimization-enabled cases use these files when Opt has prepared or revised the workspace. The runner paths below are
defaults from `CONFIGS.opt` unless `opt_space.execution` overrides them:

```text
contracts/target_spec.json
contracts/opt_space.json
contracts/default_opt_params.json
contracts/current_opt_params.json
contracts/best_opt_params.json

artifacts/opt_trials/trial_000/
artifacts/opt_trials/trial_001/
artifacts/opt_best/

reports/opt_trace.jsonl
reports/opt_report.json
reports/verification_report.json
```

### `contracts/target_spec.json`

Defines target behavior, objective terms, and success criteria.

```json
{
  "schema_version": 1,
  "task_family": "soft_ball_compression",
  "goal": {
    "final_height": 0.12,
    "final_lateral_spread": 0.24
  },
  "objective": {
    "type": "weighted_terms",
    "direction": "maximize",
    "failure_penalty": 10.0,
    "terms": [
      {
        "name": "height_match",
        "weight": -1.0,
        "metric_path": "deformation.final_height",
        "transform": "absolute_error",
        "target": 0.12
      }
    ]
  },
  "success_criteria": [
    {
      "name": "height_tolerance",
      "metric_path": "deformation.final_height_error",
      "op": "<=",
      "threshold": 0.01
    }
  ]
}
```

### `contracts/opt_space.json`

Defines bounded variables and optimization budget. Opt may create this from scratch or revise one generated by another
agent.

```json
{
  "schema_version": 1,
  "optimizer": "cma_es",
  "variables": [
    {
      "name": "control.plate_target_z",
      "type": "float",
      "default": 0.12,
      "bounds": [0.09, 0.16],
      "scale": "linear",
      "owner": "action",
      "group": "control",
      "units": "m",
      "initial_sigma": 0.2,
      "description": "Lowest target height of the press plate during compression."
    },
    {
      "name": "material.youngs_modulus",
      "type": "float",
      "default": 80000.0,
      "bounds": [10000.0, 300000.0],
      "scale": "log",
      "owner": "body",
      "group": "material",
      "units": "Pa",
      "initial_sigma": 0.35,
      "description": "FEM elastic stiffness for the soft ball."
    }
  ],
  "budget": {
    "max_trials": 24,
    "population_size": null,
    "seed": 0,
    "best_repeat_trials": 2
  },
  "strategy": {
    "early_stop": {
      "enabled": true,
      "patience_generations": 3,
      "min_delta": 0.001,
      "stop_on_success": true
    },
    "restarts": [
      {"name": "wide", "sigma_scale": 1.5, "max_trials": 8},
      {"name": "local", "sigma_scale": 0.6, "max_trials": 8, "start_from_best": true}
    ],
    "phases": [
      {"name": "control_first", "groups": ["timing", "target", "control"], "max_trials": 12},
      {"name": "contact_next", "groups": ["material", "contact"], "max_trials": 8, "start_from_best": true},
      {"name": "all_refine", "max_trials": 8, "start_from_best": true}
    ]
  }
}
```

### `contracts/default_opt_params.json`

Stores the default parameter payload used by the baseline rollout.

### `contracts/current_opt_params.json`

Written by Opt before each trial. Generated modules should read this file when present.

### `contracts/best_opt_params.json`

Written after optimization. Planner, Critic, and final renders should use this parameter set unless Planner explicitly
requests another opt pass.

## Generated Module Expectations

Generated modules should remain runnable without Opt.

Recommended behavior:

- If `contracts/current_opt_params.json` exists, read it.
- If it does not exist, fall back to `contracts/default_opt_params.json`.
- If neither exists, fall back to local safe defaults.
- Never require Opt for normal `run-suite` execution.
- Record the loaded opt params in `metrics.json`.
- Avoid stale trial params. Final renders should use `best_opt_params.json` or a copied current payload selected by Opt.

Ownership guidance:

- `scene.py` owns solver/contact lifecycle parameters.
- `body.py` owns material, geometry, initial placement, density, friction, and object parameter hooks.
- `action.py` owns policy schedule, phase timings, trajectory targets, PD gains, force limits, and control parameters.
- `assets/xml/**/*.xml` owns existing MJCF actuator/joint/geom scalar defaults when runtime Genesis APIs are not enough.
- `rendering.py` is excluded from Opt parameterization. Rendering defects should be routed back to the rendering worker.

## Validation Rules

Formal schemas live in `code_agent/specs/opt_schema/`:

- `target_spec.schema.json`
- `opt_space.schema.json`
- `opt_params.schema.json`
- `opt_trace_entry.schema.json`
- `opt_report.schema.json`
- `verification_report.schema.json`

Every variable must have:

- `name`
- `type`
- `default`
- `bounds`
- `scale`
- `owner`
- `description`

Supported in version 1:

- `type: "float"`
- `scale: "linear"`
- `scale: "log"` for positive-valued variables
- owners `scene`, `body`, `action`, or `xml`, with most variables expected in `body` or `action`
- groups `timing`, `target`, `control`, `actuator`, `initial`, `layout`, `geometry`, `material`, `contact`, `solver`,
  or `other`

Rejected in version 1:

- Unbounded variables.
- Per-step action arrays.
- Categorical variables.
- Variables that enable hidden constraints, direct target-following, or direct post-initialization object state writes.
- XML variables that require topology edits instead of scalar changes on existing elements.
- `transform: "custom"` objective terms.
- Objectives without `success_criteria` when Opt wants to claim final success.
- Variables with defaults outside bounds.
- Log-scale variables with non-positive bounds.
- Variables whose effects cannot be observed in trial metrics or visual evidence.

## Reports

### `reports/opt_trace.jsonl`

One JSON object per trial:

```json
{
  "schema_version": 1,
  "trial_index": 0,
  "status": "completed",
  "params_path": "artifacts/opt_trials/trial_000/opt_params.json",
  "artifacts_dir": "artifacts/opt_trials/trial_000",
  "metrics_path": "artifacts/opt_trials/trial_000/metrics.json",
  "score": 0.52,
  "objective": {
    "score": 0.52,
    "success": false,
    "terms": {},
    "measured": {}
  },
  "duration_sec": 18.4,
  "exit_code": 0
}
```

### `reports/opt_report.json`

Final optimization summary:

```json
{
  "schema_version": 1,
  "status": "completed",
  "optimizer": "cma_es",
  "num_trials": 24,
  "baseline_score": 0.41,
  "best_trial": 17,
  "best_score": 0.91,
  "best_params_path": "contracts/best_opt_params.json",
  "best_render_dir": "artifacts/opt_best",
  "trace_path": "reports/opt_trace.jsonl",
  "verification_report_path": "reports/verification_report.json"
}
```

### `reports/verification_report.json`

Task-oriented comparison between target and best result:

```json
{
  "schema_version": 1,
  "success": true,
  "target": {
    "final_height": 0.12
  },
  "measured": {
    "final_height": 0.126
  },
  "terms": {
    "height_match": -0.006
  },
  "score": 0.91,
  "best_trial": 17
}
```

## Available Entry Points

Planner integration is available through the `run_opt` Planner action. The action is optional and gated by the suite's
effective Opt setting, which defaults to `CONFIGS.opt.enabled` and can be overridden by `run-suite --enable-opt` or
`run-suite --disable-opt`. When disabled, Planner should not choose it. When enabled and selected, the handler invokes
`code_agent.opt.agent.run_opt_agent`, records the structured Opt result in episode state, syncs
`contracts/best_opt_params.json` to `contracts/current_opt_params.json` when Opt produces a usable result, and marks the
case for `run_execution` so Critic evaluates fresh root artifacts generated with the selected parameters.

The manual CLI remains useful for debugging the Opt agent directly:

```text
uv run python -m code_agent.opt.cli --case-dir path/to/case --backend gpu --max-rollouts 24
```

Useful debugging flags:

```text
--no-render-baseline
--no-render-best
--timeout-sec 300
--steps 300
--duration-sec 10
```

The lower-level optimization runner is still available through `code_agent.cli run-opt` after a case already has valid
Opt contracts and parameter hooks:

```text
uv run python -m code_agent.cli run-opt --case-dir path/to/case --gpu --max-trials 24
```

Useful runner flags:

```text
--no-render-best
--timeout-sec 300
--population-size 6
--seed 0
--steps 300
```

`code_agent.opt.agent.run_opt_agent` is the Planner-facing API that invokes Codex. The Planner `run_opt` action and
`code_agent.opt.cli` both call that API. The `code_agent.cli run-opt` entry point does not invoke Codex; it only runs
the numerical optimizer after the generated workspace has valid opt contracts and parameter hooks.

If the Codex Opt subagent times out or exits nonzero after writing a fresh lower-level `reports/opt_report.json`,
`run_opt_agent` recovers that report instead of returning an empty failure. A verified successful best trial becomes
`success`; an improved but still-unsuccessful best trial becomes `needs_more_optimization`; otherwise the recovered
result remains `failed` but includes the opt report, scores, variables, and evidence paths for Planner.

When the Codex Opt subagent returns `success` or `partial_success`, the harness now requires explicit best-video
evidence. If `request.render_best` is true and the report lacks a valid best video or a `video_checked=...` style
evidence item, the result is downgraded to `needs_more_optimization` so Planner does not accept a purely numeric
success claim.

## Critical Guidance

This design intentionally avoids hard-coding optimization strategy into the pipeline.

Do not make Planner or the runner assume a fixed sequence such as:

```text
baseline -> CMA-ES -> staged restart -> final render
```

Instead:

- Planner decides whether Opt is needed and what constraints matter.
- Opt decides how to parameterize and optimize based on the generated case and evidence.
- The runner and schemas provide dependable mechanics.
- Critic and Planner decide whether the result should be accepted.

The intelligence should live in the Codex agents' reasoning over code, metrics, videos, and reports. The repository
should make that reasoning executable and auditable.
