# Scripts and Suites

`code_agent/scripts/` contains suite scripts and prompt cases for the code-native pipeline.

Each category directory contains:

- `cases.txt`: `case_id|prompt` cases adapted from suite prompts and existing Genesis examples.
- `run.sh`: a wrapper that copies cases into a run directory and invokes `code_agent.cli run-suite`.

Usage:

```bash
bash code_agent/scripts/rigid_primitives/run.sh \
  --run-root code_agent/workspaces/suites/rigid_primitives/dev \
  --gpu --max-cases 1 --render
```

Useful options forwarded to `code_agent.cli run-suite`:

- `--gpu` or `--cpu`: choose backend. GPU is the default target for validation; CPU is for explicit CPU checks.
- `--max-cases N`: run a subset while iterating.
- `--max-parallel-cases N`: cap how many suite cases may run at once. The default comes from
  `CONFIGS.harness.max_parallel_cases`; `None` means all selected cases may run concurrently.
- `--render` or `--no-render`: enable or skip generated render output.
- `--duration-sec N`, `--steps N`, `--render-fps N`: override inferred simulation duration, step count, or video fps.
- `--repair-rounds N`: allow owner-routed Codex writer repair attempts after critic failure.
- `--timeout-sec N`: timeout for each generated simulation.
- `--enable-deformable` or `--disable-deformable`: override `CONFIGS.deformable.enabled`. Deformable generation is
  disabled by default and must be explicitly enabled for FEM+IPC primitive work.

Suite cases run concurrently by default because each case owns an independent workspace. The local Genesis execution
stage is still serialized by a per-user lock in `utils/execution.py`, so only one generated simulation process runs at a
time even while Planner, asset, writer, and critic work overlaps across cases.

The planner and all four writer modules run for every suite case.
Suite startup also fetches or reuses the selected official Genesis documentation context for FEM+IPC scenes. To build
that context without running a suite:

```bash
uv run python -m code_agent.cli build-genesis-context \
  --out-dir code_agent/workspaces/context_smoke
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
- `examples/IPC_Solver/ipc_objects_falling.py`
- `examples/IPC_Solver/ipc_robot_grasp_cube.py`

The active non-rigid target for new code-agent work is FEM+IPC. Other Genesis non-rigid families are not included in
the context pack unless they are explicitly reintroduced later.

Run deformable primitive cases with deformable generation explicitly enabled, for example:

```bash
bash code_agent/scripts/deformable_primitives/run.sh \
  --enable-deformable --gpu --render --max-cases 1
```

FEM+IPC runs are expected to be much slower than rigid-only cases, especially with stacked soft bodies and Genesis
camera rendering. Low wall-clock frame throughput or multi-minute execution is not by itself a failure. Treat a run as
broken only when it times out, crashes, stops making progress for a long period, produces invalid physics artifacts
such as NaNs/explosions, or the evaluator/critic identifies a concrete source or output problem.

For FEM elastic bodies, generated code should pass explicit `E`, `nu`, and `rho` values. The numeric defaults and ranges
come from `CONFIGS.deformable`, while the shared prompt guide explains their physical meaning for agent-side selection.

When deformable generation is disabled, Planner should stop soft-body tasks as inconclusive instead of generating a
rigid-body substitute.

### `scripts/deformable_mesh/`

Script:

- `code_agent/scripts/deformable_mesh/run.sh`

Cases:

- `examples/elastic_dragon.py`
- `examples/coupling/cut_dragon.py`
- `examples/coupling/grasp_soft_cube.py`
- `examples/sap_coupling/franka_grasp_fem_sphere.py`

## Execution Rule

Future scripts that invoke Python, `uv`, `pytest`, or Genesis should run through the repository uv environment and use
the dedicated local GPU by default. Use CPU only for explicit CPU checks or when GPU execution is unavailable.

The scripts call `uv run python -m code_agent.cli run-suite` directly.

For mesh suites, make sure `MESHY_API_KEY` is exported in the same non-interactive shell that launches the script.
This repository's usual `~/.bashrc` may return before later API-key exports when sourced by non-interactive commands,
so `source ~/.bashrc` alone is not a reliable check. Load or export the key explicitly without echoing its value.
