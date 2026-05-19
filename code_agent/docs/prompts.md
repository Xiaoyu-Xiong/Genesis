# Prompts

`code_agent/prompts/` owns shared prompt text and prompt builders for Codex calls. Runtime utilities should import from
the concrete modules in this directory instead of carrying prompt contracts inside `code_agent/utils/`. The directory
does not provide top-level exports; import from `code_agent.prompts.common`, `code_agent.prompts.planner`, and so on.

- `common.py`: shared physical causality, scale, rendering clarity, and source-aware repair guidance.
- `ipc.py`: FEM material, rigid-IPC coupling, external MJCF articulation, and IPC failure-diagnostic guidance.
- `planner.py`: Planner prompt rules, action policy, and rollout/repair guidance.
- `worker.py`: writer prompt rules plus rigid and FEM API guidance.
- `critic.py`: evaluation and critic prompt guidance.
- `opt.py`: Opt Codex subagent role prompt and request builder.

Keep this package focused on reusable prompt material. Agent orchestration, execution, schema validation, and file-system
mechanics should remain in their existing Planner, writer, evaluation, opt, or utils modules.
