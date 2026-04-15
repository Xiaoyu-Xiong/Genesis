# IR and Runtime

This document describes the current IR structure, body semantics, action semantics, runtime execution, and event-pack output.

## IR Root

The root IR uses top-level `bodies`.

Current high-level structure:

- `scene`
- `bodies`
- `actions`

Old forms such as top-level `body` or body-level `static` are not accepted.

## Bodies

Each body currently includes:

- `name`
- `shape`
- `initial_pose`
- `fixed`
- `collision`
- optional `rho`
- `simulation_kind`
- optional `deformable_material`
- optional `actuators`

### Shape Kinds

Supported shape kinds:

- primitive: `sphere`, `box`, `cylinder`
- mesh: `mesh`
- articulated: `mjcf`, `urdf`

### Simulation Kind

`bodies[].simulation_kind` controls whether the body is:

- `rigid`
- `deformable`

For deformable bodies, the currently allowed geometry is:

- `sphere`
- `box`
- `cylinder`
- `mesh`

The active deformable backend and its system-level hyperparameters are defined in [configs.py](../configs.py).

### Fixed Bodies

Use `fixed: true` for:

- rigid obstacles
- platforms
- anchored rigid mesh props
- fixed-base URDF bodies

Important limitation:

- `mjcf` bodies do not accept body-level `fixed`
- fixed-base MJCF behavior must be encoded inside the XML

### Mesh Shape

`shape.kind="mesh"` is used for non-articulated imported or generated mesh bodies.

Important fields:

- `file`
- `scale`

`scale` is the uniform whole-mesh scale factor.
Use it when the mesh is globally too large or too small for the scene.

### Density and Soft Material Notes

Current validation enforces:

- all `rho` values must stay in `[300, 3000]`

For deformable FEM scenes, the main task-level material fields are:

- `rho`
- `E`
- `nu`

For rigid bodies, `rho` changes mass and inertia but not geometric size.

## Actions

Core actions include:

- `step`
- `observe`
- `set_pose`
- `apply_external_wrench`
- `set_dofs_position`
- `set_dofs_velocity`
- `set_target_pos`
- `set_torque`

Action targeting uses `entity`.

`observe`, `set_pose`, and `apply_external_wrench` may target:

- a single body name
- a list of body names

Actuator and direct DoF actions remain single-entity.

For deformable bodies:

- do not use actuator actions on deformables
- prefer deformable-friendly observation fields such as `bbox_*` and displacement statistics

## Runtime

[agent/cli.py](../cli.py) provides:

- `schema`
- `validate`
- `compile`
- `run`

Examples:

```bash
uv run python -m agent.cli validate \
  --ir path/to/ir.json \
  --out /tmp/ir.validated.json

uv run python -m agent.cli run \
  --ir /tmp/ir.validated.json \
  --out /tmp/run_result.json \
  --event-pack-out /tmp/event_pack.json

uv run python -m agent.cli compile \
  --ir /tmp/ir.validated.json \
  --out /tmp/compiled_genesis.py
```

## Event Pack

`agent.cli run` can emit an LLM-facing structured event pack.

Top-level event-pack structure includes:

- `scene`
- `entities`
- `execution`
- `action_trace`
- `observations`
- `highlights`

Useful observation indexes:

- `observations.by_entity_indices`
- `observations.by_entity_tag_indices`
- `observations.by_entity_last_observation`
