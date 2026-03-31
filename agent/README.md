# Agent Rigid-Scene Pipeline

This directory contains the IR, runtime, compiler, generator, critic, and optimization loop used to turn natural-language tasks into Genesis rigid-scene simulations.

The current system supports:

- multiple primitive bodies
- multiple articulated bodies (`mjcf` or `urdf`)
- body-level `fixed` support for primitives and URDF
- multi-entity actions for `observe`, `set_pose`, and `apply_external_wrench`
- direct execution, code generation, LLM generation, multimodal critique, and iterative optimization

The public IR class name is `RigidIR`, and the root structure uses top-level `bodies`.
Old payload forms such as top-level `body` or body-level `static` are no longer accepted.

## Directory Layout

- `ir_schema/`: pydantic IR models and validation
- `runtime/`: direct Genesis execution and event-pack generation
- `compiler_backend/`: `IR -> generated Genesis Python`
- `tool_library/`: shared tool specs and validation logic used by generator and critic
- `llm_generator/`: two-agent IR/XML generation
- `llm_critic/`: multimodal critique from IR + event pack + video
- `opt/`: generator -> run -> critic -> feedback refinement loop
- `scripts/`: batch scripts for common benchmark suites
- `generated_assets/`: generated MJCF assets
- `runs/`: run artifacts, videos, logs, critic outputs

## IR Overview

The IR is a rigid-scene program:

- `scene`: backend, timestep, gravity, ground, viewer, render config
- `bodies`: all scene bodies
- `actions`: ordered action program

### Bodies

Each body has:

- `name`
- `shape`
- `initial_pose`
- `fixed`
- `rho`
- `collision`
- optional `actuators`

Supported shape kinds:

- primitive: `sphere`, `box`, `cylinder`
- articulated/imported: `mjcf`, `urdf`

Current articulated policy:

- multiple primitive bodies are allowed
- multiple articulated bodies are allowed

### Fixed Bodies

Use `fixed: true` for:

- static obstacles
- tables
- platforms
- static targets
- anchored URDF bodies

Important limitation:

- `mjcf` bodies do not accept body-level `fixed`
- for MJCF, a fixed base must be expressed inside the XML itself, for example by not using a free joint

### Actions

Core actions:

- `step`
- `observe`
- `set_pose`
- `apply_external_wrench`
- `set_dofs_position`
- `set_dofs_velocity`
- `set_target_pos`
- `set_torque`

Action targeting uses `entity`.

Single-entity actions:

- all actions accept a single body name

Multi-entity actions:

- `observe`
- `set_pose`
- `apply_external_wrench`

For these actions, `entity` can be either:

- a single body name
- a list of body names

Use the list form when the same payload should be broadcast to multiple bodies at the same timestep. This keeps IR shorter and easier to critique.

### Actuator Control

Actuator control is scoped by body:

- `set_target_pos` addresses a position actuator on `action.entity`
- `set_torque` addresses a motor actuator on `action.entity`

Actuator and DOF control actions remain single-entity only.

### External Wrench Semantics

`apply_external_wrench` is an external disturbance action, not an actuator command.

Typical usage pattern:

1. set nonzero force and/or torque
2. `step` for some duration
3. write the wrench back to zero

The effect persists across subsequent `step` actions until another wrench update changes it.

## Event Pack

`agent.cli run` can emit an LLM-friendly structured event pack.

Top-level structure includes:

- `scene`
- `entities`
- `execution`
- `action_trace`
- `observations`
- `highlights`

Use `entities` as the top-level body summary in the event pack.

Useful observation indexes:

- `observations.by_entity_indices`
- `observations.by_entity_tag_indices`
- `observations.by_entity_last_observation`

## CLI

### Schema

```bash
uv run python -m agent.cli schema --out /tmp/schema.json
```

### Validate

```bash
uv run python -m agent.cli validate \
  --ir path/to/ir.json \
  --out /tmp/ir.validated.json
```

### Compile

```bash
uv run python -m agent.cli compile \
  --ir path/to/ir.validated.json \
  --out /tmp/compiled_genesis.py
```

### Run

```bash
uv run python -m agent.cli run \
  --ir path/to/ir.validated.json \
  --out /tmp/run_result.json \
  --event-pack-out /tmp/event_pack.json
```

## LLM Generator

The generator uses a two-agent flow:

- IR agent: plans and validates IR through tool-calling
- XML agent: generates MJCF only when needed

The generator sees:

- IR schema
- generation guide
- observation-field guide
- validation tool
- parameter notes and relationship notes

It is explicitly instructed to:

- use top-level `bodies`
- keep IR concise
- use multi-entity actions when payload and timing are identical
- use actuator-driven motion for articulated bodies
- avoid direct state writes after simulation starts

Example:

```bash
uv run python -m agent.llm_generator.cli generate \
  --task "Create a fixed platform, a movable box, and a sphere dropped onto the box over 6 seconds." \
  --model gpt-5.4 \
  --reasoning-effort high \
  --assets-dir agent/generated_assets \
  --out /tmp/generated_ir.json \
  --log-out /tmp/generation.log.json
```

Hosted Prompt support is available through:

- `--hosted-prompt-id`
- `--hosted-prompt-version`

## LLM Critic

The critic evaluates:

- task prompt
- IR
- optional XML
- event pack
- sampled video frames

It returns a structured critique with:

- `verdict`
- `overall_score`
- `summary`
- `by_section.scene`
- `by_section.actions`
- `by_body`
- `priority_fixes`

The critic is instructed to:

- stay within generator/tool-library capabilities
- use schema descriptions and parameter notes
- focus on main problems instead of minor numeric noise
- prefer concise IR when behavior is unchanged

Example:

```bash
uv run python -m agent.llm_critic.cli evaluate \
  --task "Create a newton cradle with five suspended equal spheres." \
  --ir path/to/ir.validated.json \
  --xml robot_a=path/to/robot_a.xml \
  --xml robot_b=path/to/robot_b.xml \
  --event-pack path/to/event_pack.json \
  --video path/to/render.mp4 \
  --model gpt-5.4 \
  --reasoning-effort high \
  --out /tmp/critic.json \
  --log-out /tmp/critic.log.json
```

If critic evaluation fails upstream, the optimization loop may emit a synthetic fallback `critic.json`. In that case, inspect `critic.log.json` before trusting the score.

## Optimization Loop

`agent.opt.cli optimize` runs:

1. generate IR
2. validate and normalize
3. execute simulation
4. build event pack and render video
5. critique with the multimodal critic
6. feed structured feedback into the next round

Current feedback behavior:

- later rounds receive the previous validated IR
- later rounds receive previous XML text when applicable
- XML-specific feedback is routed separately
- generator is told to revise the previous result instead of regenerating from scratch

Example:

```bash
uv run python -m agent.opt.cli optimize \
  --task "Create a fixed box platform, a movable box, and a sphere dropped onto the box over 6 seconds." \
  --model gpt-5.4 \
  --backend gpu \
  --reasoning-effort high \
  --out-dir agent/runs/example_opt \
  --out agent/runs/example_opt/summary.json
```

## Scripts

Useful entrypoints under `agent/scripts/`:

- `run.sh`: small convenience runner
- `run_opt_robot_suite.sh`: articulated/robot-heavy optimization suite
- `run_opt_multibody_suite.sh`: multi-body suite with classic rigid-body scenes

These scripts:

- set `PYTHONPATH` to the repo root
- prefer `.venv/bin/python` when present
- fall back to `uv run python` otherwise

## Key Constraints and Notes

- IR class name is `RigidIR`, and the root IR field is `bodies`
- current support is multiple primitive and articulated bodies
- `fixed=true` is supported for primitives and URDF, not for MJCF
- `set_pose`, `set_dofs_position`, and `set_dofs_velocity` are intended for pre-simulation use only
- `kp`, `kv`, and `force_range` should be interpreted together:
  - `kp`: tracking stiffness
  - `kv`: damping
  - `force_range`: output limit
- `rho` changes mass and inertia, not geometric size
- `friction` increases resistance to sliding, but does not guarantee perfect sticking

## Suggested Workflow

For hand-authored IR:

1. write IR JSON
2. validate
3. run
4. inspect `event_pack.json`
5. optionally compile to generated Genesis code

For natural-language generation:

1. generate IR with `agent.llm_generator.cli`
2. validate and run
3. critique with `agent.llm_critic.cli`
4. optimize with `agent.opt.cli` when iterative repair is needed
