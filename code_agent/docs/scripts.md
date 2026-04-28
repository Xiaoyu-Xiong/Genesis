# Scripts and Suites

`code_agent/scripts/` contains first-pass suite scripts and prompt cases for the code-native pipeline.

Each category directory contains:

- `cases.txt`: `case_id|prompt` cases copied or adapted from legacy `agent/scripts` suites and existing Genesis examples.
- `run.sh`: a wrapper that copies cases into a run directory and invokes the future `code_agent.cli run-suite` entrypoint.

The `code_agent` CLI is not implemented yet. Until it exists, `run.sh` exits with a clear message unless `CODE_AGENT_CMD`
is set to an experimental command.

Usage shape:

```bash
code_agent/scripts/rigid_primitives/run.sh --run-root code_agent/workspaces/suites/rigid_primitives/dev
```

Override command example:

```bash
CODE_AGENT_CMD="apptainer exec --nv /ocean/projects/cis250078p/xxiong1/containers/genesis.sif uv run python -m code_agent.cli run-suite" \
  code_agent/scripts/rigid_primitives/run.sh
```

## Categories

### `scripts/rigid_primitives/`

Script:

- `code_agent/scripts/rigid_primitives/run.sh`

Cases:

- `tests/test_rigid_physics.py`
- `tests/test_rigid_benchmarks.py`
- `examples/collision/pyramid.py`
- `examples/collision/tower.py`
- `examples/rigid/apply_external_force_torque.py`

### `scripts/rigid_articulated/`

Script:

- `code_agent/scripts/rigid_articulated/run.sh`

Cases:

- `examples/rigid/single_franka.py`
- `examples/rigid/control_franka.py`
- `examples/rigid/ik_franka.py`
- `examples/tutorials/control_your_robot.py`
- `tests/test_kinematic.py`

### `scripts/rigid_mesh/`

Script:

- `code_agent/scripts/rigid_mesh/run.sh`

Cases:

- `tests/test_mesh.py`
- `examples/rigid/control_mesh.py`
- `examples/rigid/nonconvex_mesh.py`
- `examples/rigid/terrain_from_mesh.py`

### `scripts/deformable_primitives/`

Script:

- `code_agent/scripts/deformable_primitives/run.sh`

Cases:

- `tests/test_deformable_physics.py`
- `tests/test_fem.py`
- `examples/fem_hard_and_soft_constraint.py`
- `examples/sap_coupling/fem_sphere_and_cube.py`
- `examples/IPC_Solver/ipc_objects_falling.py`

### `scripts/deformable_mesh/`

Script:

- `code_agent/scripts/deformable_mesh/run.sh`

Cases:

- `examples/elastic_dragon.py`
- `examples/coupling/cut_dragon.py`
- `examples/coupling/grasp_soft_cube.py`
- `examples/sap_coupling/franka_grasp_fem_sphere.py`

## Execution Rule

Future scripts that invoke Python, `uv`, `pytest`, or Genesis must run inside Apptainer or through approved sbatch
execution.
