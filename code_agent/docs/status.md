# Status

The active pipeline is Planner-led and code-native.

Implemented:

- suite CLI and per-case workspaces
- Planner action loop with persisted state
- background mesh and XML/MJCF asset jobs
- ready asset inspection previews for mesh/XML debugging
- parallel Scene, Body, Action, and Rendering writer dispatch
- deterministic integration into `src/main.py`
- local uv/GPU execution with serialized Genesis subprocesses
- deterministic artifact checks, visual evidence, and single-pass Codex Critic
- owner-routed repair for writer-owned failures
- ordinary rigid, rigid+IPC, and FEM+IPC capability contracts

Current limits:

- worker write-scope validation is structural, not a full git diff audit
- mesh/XML assets can still fail provider, repair, import, or actuator validation and should be regenerated through
  Planner asset actions
- FEM+IPC runs can be slow; use logs, metrics, and critic evidence before treating low throughput as failure

Keep this file factual and short. Detailed history belongs in git, not docs.
