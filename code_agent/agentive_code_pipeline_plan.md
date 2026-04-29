# Codex-First Code Agent Pipeline Plan

本文档记录当前 `code_agent/` 的主动规划方向。当前实现不再维护旧的手工串联主循环；单个例子的生命周期
由 Planner-led episode runtime 驱动。

## Current Architecture

`utils/suite.py` 只负责读取 suite case、创建 workspace、写入输入文件，并为每个 case 启动一个
`PlannerSession`。之后的 case 级控制权交给 Planner。

`PlannerSession` 和 Planner 调度层位于 `code_agent/planner/`：

- `planner/agent.py`: Planner prompt 构造、`codex exec` 调用、Planner action 读取。
- `planner/session.py`: episode 状态机、持久化、summary、writer 状态记录。
- `planner/actions.py`: 执行 Planner 选择的 harness action。

负责写代码的 subagent 位于 `code_agent/writer/`。`code_agent/utils/` 保留 suite 入口、Codex 调用、本地执行、
timing resolution 和生成入口文件的 integration helper。

Planner 每一轮只返回一个符合 `planner_action.schema.json` 的结构化 action。Python harness 负责执行这个
action、更新 `reports/episode_state.json`、追加 action/dispatch JSONL，然后再次调用 Planner，直到 Planner
选择 `finish` 或预算耗尽。

## Planner Action Library

当前暴露给 Planner 的动作库是固定的、窄接口的：

- `write_plan`: 写入 `planner_output`，由 harness 校验 schema 并解析 duration、steps、fps。
- `spawn_workers`: 唤醒 Scene、Body、Action、Rendering 中的一个或多个生成 worker。
- `run_integrator`: 写入稳定的 `src/main.py`。
- `run_execution`: 使用仓库 uv 环境和本机默认 GPU 运行生成代码。
- `run_critic`: 运行 deterministic checks 和 Codex Critic。
- `request_repair`: 把失败上下文发回指定 owner worker。
- `run_python`: 运行受控 `uv run python ...`。
- `run_pytest`: 运行受控 `uv run pytest ...`。
- `finish`: 结束 episode，输出 pass、fail 或 inconclusive。

Planner 不直接编辑文件，也不直接执行 shell 命令。所有文件写入、GPU 使用、命令执行、schema 校验、
artifact 收集和 retry 预算都由 harness 执行。

## Worker Ownership

当前生成 worker 仍然是四个：

- Scene Worker: 场景、固定物体、全局模拟设置、可供渲染参考的空间锚点。
- Body Worker: 可动刚体、任务参与物体、actor 字典。
- Action Worker: step loop、控制、指标、event log、任务结果。
- Rendering Worker: Genesis camera、相机参数、灯光、帧采样、视频合成、render stats。

worker 在 `workspace-write` sandbox 中运行，但只能编辑自己被分配的目标文件。dispatcher 会检查
`worker_report.schema.json`、changed files、目标文件存在性和必需 export。

## Timing Policy

时间相关参数由 Planner 通过 `planner_output.execution_plan` 明确给出。`utils.timing` 不再用正则或关键字从
自然语言里推断时长、step 数或 fps。

CLI 的 `--duration-sec`、`--steps`、`--render-fps` 是显式 override。没有 override 时，harness 使用 Planner
输出的 duration、step budget 和 render fps。默认运行目标是本机 GPU。

## Execution And Evaluation

utils execution 层只做本地 uv 执行和 artifact 收集。它不判断任务质量，也不生成代码。

当前 runner 写入：

- `reports/execution_report.json`
- `reports/stdout.txt`
- `reports/stderr.txt`
- 发现到的 artifact 路径

evaluation 层先做 deterministic checks，再调用 Codex Critic。Critic 是只读调用，读取 execution report、
metrics、event log、render stats 等证据，返回结构化 verdict、score、recommended owner 和 repair summary。

## Artifacts

每个 case workspace 的核心输出是：

- `contracts/planner_output.json`
- `contracts/timing.json`
- `contracts/episode_plan.json`
- `src/scene.py`
- `src/body.py`
- `src/action.py`
- `src/rendering.py`
- `src/main.py`
- `reports/episode_state.json`
- `reports/planner_actions.jsonl`
- `reports/dispatch_history.jsonl`
- `reports/execution_report.json`
- `reports/artifact_evaluation.json`
- `reports/critic_report.json`
- `reports/codex_critic_report.json`
- `artifacts/run_result.json`
- `artifacts/event_log.json`
- `artifacts/metrics.json`
- `artifacts/render_stats.json`
- `artifacts/render.mp4`
- `summary.json`

## Current Defaults

- 使用仓库 uv 环境直接运行。
- GPU 是默认执行目标。
- CPU 只用于显式 CPU 请求、GPU 不可用或明确 CPU-only 的任务。
- Planner 和 Critic 使用 read-only sandbox。
- 生成 worker 使用 workspace-write sandbox。
- 不使用 Apptainer 或 Slurm。

## Near-Term Work

后续结构性工作应围绕当前 Planner action library 扩展，而不是恢复旧手工 pipeline：

- 增加 episode resume。
- 增加 worker 写入范围的 diff audit。
- 接入 Asset Bridge。
- 接入 mesh asset request。
- 接入 Codex XML/MJCF worker。
- 扩大 rigid primitive suite 覆盖并验证重复运行稳定性。
