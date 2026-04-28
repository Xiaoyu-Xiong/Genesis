# Execution

`code_agent/execution/` is the future home of generated-simulation execution and artifact collection.

## Responsibilities

- Run generated code inside Apptainer for local smoke execution.
- Submit GPU-heavy or long runs through approved sbatch paths.
- Collect stdout, stderr, exit status, timing, and resource usage.
- Collect required artifacts such as metrics, event logs, video, frames, and run summaries.
- Emit `execution_report.json`.

## Boundaries

- Do not generate or edit simulation code here.
- Do not run Python, `uv`, `pytest`, or Genesis outside Apptainer.
- Do not make task-quality judgments; pass artifacts to [Evaluation](evaluation.md).
