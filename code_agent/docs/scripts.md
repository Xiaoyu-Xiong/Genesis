# Scripts and Suites

`code_agent/scripts/` contains first-pass suite scripts and prompt cases for the code-native pipeline.

Each category directory contains:

- `cases.txt`: `case_id|prompt` cases copied or adapted from legacy `agent/scripts` suites and existing Genesis examples.
- `run.sh`: a wrapper that copies cases into a run directory and invokes `code_agent.cli run-suite`.

Usage:

```bash
apptainer exec /ocean/projects/cis250078p/xxiong1/containers/genesis.sif \
  bash code_agent/scripts/rigid_primitives/run.sh \
  --run-root code_agent/workspaces/suites/rigid_primitives/dev \
  --cpu --codex-mode off --generation-mode codex --max-cases 1 --no-render
```

Useful options forwarded to `code_agent.cli run-suite`:

- `--cpu` or `--gpu`: choose backend. The current smoke validation uses `--cpu`.
- `--codex-mode off|auto|required`: disable planner, record planner/fallback, or require planner success.
- `--generation-mode codex|fallback`: choose Codex writer generation or the deterministic fallback generator.
- `--max-cases N`: run a subset while iterating.
- `--render` or `--no-render`: enable or skip generated render output.
- `--repair-rounds N`: allow owner-routed Codex writer repair attempts after critic failure.
- `--timeout-sec N`: timeout for each generated simulation.

`--codex-mode` currently controls only the planner adapter. It does not select writer generation; use
`--generation-mode` for that.

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

The scripts detect when they are already inside Apptainer and then call `uv run python -m code_agent.cli run-suite`
directly. From the host, they wrap the CLI with `apptainer exec` using the standard Genesis image.
