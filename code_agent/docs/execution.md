# Execution

`code_agent/execution/` runs generated simulations and collects artifacts.

## Responsibilities

- Run generated code inside Apptainer for local smoke execution.
- Submit GPU-heavy or long runs through approved sbatch paths.
- Collect stdout, stderr, exit status, timing, and resource usage.
- Collect required artifacts such as metrics, event logs, video, frames, and run summaries.
- Emit `execution_report.json`.

## Current MVP Behavior

`runner.py` adapts generated projects to `apptainer_cpu.py`. When invoked from inside Apptainer, the runner executes
`uv run python src/main.py --backend cpu --out-dir artifacts --steps 40 --render/--no-render` directly in the case
workspace. When invoked from the host, `apptainer_cpu.py` wraps the generated command in `apptainer exec` using the
standard Genesis image.

The report is written to `reports/execution_report.json` and includes command metadata, exit status, timeout status,
stdout/stderr paths, and discovered artifacts. A compatibility `reports/legacy_execution_report.json` is also written
for early orchestration code.

Execution does not decide how rendering is implemented. It only passes render-related CLI flags to generated code and
collects artifacts such as `render.mp4`, `frames/`, and `render_stats.json` when they exist. Camera placement, render
cadence, renderer fallback, and visual debugging logic belong to the Rendering Worker.

## Boundaries

- Do not generate or edit simulation code here.
- Do not run Python, `uv`, `pytest`, or Genesis outside Apptainer.
- Do not make task-quality judgments; pass artifacts to [Evaluation](evaluation.md).
