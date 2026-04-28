# Codex-First Code Agent Pipeline Plan

本文档规划新的 `code_agent/` 管线：不再以 IR 作为主生成路径，而是从 text prompt 直接生成 Genesis 仿真代码。第一版原则是最大化复用 Codex CLI，避免重新实现 agent 框架、代码编辑器、reviewer、debugger 和 repo 浏览能力。

核心判断：

- Codex 负责规划、写代码、集成、review、debug。
- `code_agent` 只负责薄调度、资产桥接、执行控制、artifact 管理和验证闭环。
- 生成代码是最终权威产物；Scene Brief、Module Contract、Asset Manifest 等只作为协作元数据，不是新的 IR。

## 1. Design Constraints

### 1.1 Goals

- 从自然语言任务生成可运行的 Genesis Python 仿真工程。
- 支持复杂场景分解，而不是要求单个模型一次生成完整脚本。
- 复用迁移后的 [mesh](docs/mesh.md) 和资产 post-processing 能力；MJCF/XML 由 Codex asset worker 生成或从 repo 资产引用。
- 通过 [configs.py](configs.py) 固化通用默认超参，减少 Codex agents 在常规参数上的决策负担。
- 保证所有 Python / `uv` / `pytest` / Genesis 执行仍由 Apptainer 或 sbatch 统一控制。
- 保存可审计 artifact：prompt、planner output、worker logs、代码、执行日志、视频、metrics、critic 报告。

### 1.2 Non-Goals

- 不在第一版自建 LLM client。
- 不自建通用 multi-agent conversation framework。
- 不让顶层 planner 任意递归启动子 agent。
- 不让 Codex worker 自行运行 Genesis 仿真。
- 不重写 mesh repair、texture transfer 等现有 mesh 后处理能力。
- 不照搬旧 XML agent；MJCF/XML 按 code-native 思路交给 Codex subagent 生成、review 和修复。
- 不把 Scene Brief / Module Contract 发展成新的可执行 IR。

## 2. Architecture

### 2.1 Layer Split

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Control Plane | `code_agent` deterministic coordinator | workspace、schema validation、Codex invocation、write-scope enforcement、asset bridge、Apptainer/sbatch execution、retry budget |
| Agent Plane | Codex CLI | planning、code generation、integration、review、debug、critic reasoning |
| Asset Plane | existing mesh code plus Codex XML worker | Meshy、mesh repair、texture transfer、Codex-generated MJCF/XML、repo asset lookup |
| Evaluation Plane | mixed | artifact completeness、task metrics、video/event critic、suite summaries |

关键边界：Codex 可以写代码和提出修复，但生成代码的执行必须由 Control Plane 统一完成。

### 2.2 Repository Skeleton

The scaffold is intentionally compact. Subdirectory documentation is centralized under [docs](docs/README.md):

| Directory | Documentation |
| --- | --- |
| `orchestration/` | [Orchestration](docs/orchestration.md) |
| `codex/` | [Codex](docs/codex.md) |
| `execution/` | [Execution](docs/execution.md) |
| `evaluation/` | [Evaluation](docs/evaluation.md) |
| `specs/` | [Specs](docs/specs.md) |
| `assets/` | [Assets](docs/assets.md) |
| `assets/mesh/` | [Migrated Mesh Pipeline](docs/mesh.md) |
| `scripts/` | [Scripts and Suites](docs/scripts.md) |
| `workspaces/` | [Workspaces](docs/workspaces.md) |
| `docs/` | [Documentation Index](docs/README.md) |

### 2.3 Top-Level Flow

1. `code_agent` CLI 接收用户 prompt、run id、资源选项和输出目录。
2. Coordinator 创建 run workspace，写入用户任务、仓库规则、能力说明。
3. Coordinator 调用 read-only Codex Planner，生成结构化 `planner_output.json`。
4. Coordinator 校验 planner output，拆分 Scene Brief、Asset Requests、Module Contracts。
5. Coordinator 调用 Asset Bridge 生成或查找所需资产：mesh 走迁移后的 [mesh](docs/mesh.md) pipeline，MJCF/XML 走 Codex XML worker，输出 `asset_manifest.json`。
6. Coordinator 固定派发 Scene / Body / Action 三个 Codex workers 生成主仿真模块。
7. Codex Integrator 合并入口、导入、函数签名和 artifact 输出。
8. Codex Review Agent 或 `codex exec review` 做静态审查。
9. Execution Agent 用 Apptainer 或 sbatch 运行生成代码。
10. Coordinator 收集 run artifacts、metrics、video、stdout/stderr。
11. 单层 Critic Agent 结合任务、metrics、event log 和视频给出 verdict。
12. 若失败，Debugger 输出 owner-routed patch plan，Coordinator 调用目标 Codex worker 修复。
13. 成功或达到重试上限后写 `summary.json`。

### 2.4 Codex Invocation Policy

Planner / reviewer 默认只读：

```bash
codex exec \
  --cd /jet/home/xxiong1/Genesis \
  --sandbox read-only \
  --ask-for-approval never \
  --json \
  --output-last-message code_agent/workspaces/<run_id>/logs/<role>.final.md \
  --output-schema code_agent/schemas/<role_output>.schema.json \
  "<role prompt>"
```

Writer / integrator / debugger 可写 workspace：

```bash
codex exec \
  --cd /jet/home/xxiong1/Genesis \
  --sandbox workspace-write \
  --ask-for-approval never \
  --json \
  --output-last-message code_agent/workspaces/<run_id>/logs/<role>.final.md \
  --output-schema code_agent/schemas/<role_output>.schema.json \
  "<role prompt>"
```

每个 writer prompt 必须包含：

- 角色职责
- 允许修改的文件路径
- 禁止运行 host-side Python / `uv` / `pytest` / simulation
- 输入 contracts
- Asset Manifest
- 期望导出的函数或入口
- final report schema

Coordinator 必须检查 worker report 和 git/workspace diff，确认修改没有越界。

### 2.5 Static Configs

[configs.py](configs.py) 保存 `code_agent` 第一版的静态默认超参。它从旧 `agent/configs.py` 迁移了和 IR 无关、通常不需要 agent 动态修改的参数：

- `CodexConfigs`: planner、worker、reviewer、debugger、critic 的默认模型、sandbox 和 approval policy。
- `OrchestrationConfigs`: coordinator 并发、重试、timeout、backend、scope check 和 static review 开关。
- `RuntimeConfigs`: Genesis timestep、substeps、render interval 和 resolution。
- `DeformableConfigs`: FEM+IPC 物理默认值；PBD 暂不进入 code-agent 第一版。
- `CriticConfigs`: 单层 critic 的 video sampling、frame count 和 frame width 默认值。
- `MeshyRequestConfigs`: Meshy 请求默认值。
- `MeshRepairConfigs`: repair、fTetWild、texture transfer 默认值。

这些配置是硬编码默认值，不应从环境变量隐式加载。Run-specific 行为应通过明确 CLI flag 或 run config 覆盖。Codex workers 应优先使用这些默认值，除非用户任务、执行失败或 critic 反馈明确要求调整。

## 3. Contracts

### 3.1 Planner Output

Codex Planner 是第一版的顶层智能体。它只做规划，不改代码，不直接启动子 agent。

输出字段：

- `scene_brief`: 用户意图、必须实体、交互目标、成功/失败标准、默认假设。
- `scene_plan`: 仿真策略、物理风险、资源级别、渲染需求。
- `asset_requests`: 资产名称、类型、用途、尺度、bbox、纹理需求、仿真角色。
- `module_contracts`: worker role、target files、exports、dependencies、forbidden edits。
- `dispatch_graph`: 可并行 worker、依赖关系、是否等待 asset manifest。
- `execution_plan`: local Apptainer smoke test 或 GPU sbatch、预计 step/render budget。
- `risk_register`: 主要失败模式和预防策略。

### 3.2 Module Contract

每个模块 contract 至少包含：

- owner role
- target files
- allowed write paths
- required exports
- input dependencies
- asset dependencies
- forbidden edits
- smoke expectation
- final report schema

第一版固定使用一条主仿真拆分路径，不再按“简单/复杂”分支。拆分沿用旧 IR 的直觉，但改成代码模块职责：

- `Scene Worker`: 搭建舞台。负责 scene lifecycle、ground、arena、camera、lights、fixed obstacles、fixed props、static supports、global FEM+IPC defaults 和 artifact layout。常规 dt/substeps/material/contact 参数从 [configs.py](configs.py) 读取。固定物体属于 scene，不属于 body。
- `Body Worker`: 定义演员。只负责会运动、会被驱动、会被接触推移、会变形或参与任务结果的实体，包括 dynamic rigid bodies、deformables、robots、free-base MJCF/URDF、active tools 和 movable mesh assets。
- `Action Worker`: 编写剧本。负责 staged control、robot DoF schedule、external forces、scripted policy、event logging、task metric、failure guards、render trigger 和 final score。
- `Integrator`: 合并 Scene / Body / Action，维护 imports、entrypoint、module wiring 和最终 runnable project。

这个拆分足够粗，避免过多 subagent 协调；同时又比单一 writer 更利于 owner-routed repair。

### 3.3 Asset Manifest

资产 agent 不直接写主仿真代码，只输出可被代码消费的 manifest。

每个资产记录：

- logical name
- source type: primitive / generated_mesh / repo_asset / mjcf / urdf
- runtime path
- visual path
- texture path
- bbox
- recommended scale
- physical role: visual only / collision / deformable / rigid / articulated
- validation status
- known caveats
- suggested Genesis construction pattern

主仿真 worker 只能使用 manifest 中标为 runtime-ready 的路径，不能猜测 Meshy 输出目录，也不能把 raw textured OBJ 当作 runtime mesh。

## 4. Asset Bridge

第一版资产层复用迁移后的 mesh 能力，并用 Codex worker 处理 MJCF/XML：

- [mesh](docs/mesh.md): Meshy preview、texture refine、manifold check、repair、texture transfer、validation render。这个路径从旧 `agent/mesh` 迁移到 `assets/mesh`，只在外层增加 request normalization 和 manifest 汇总。
- Codex XML worker: articulated asset 的 MJCF/XML 由专门的 Codex subagent 从头生成、review、修复，并登记到 Asset Manifest。每个 XML 只允许包含一个 articulated body，不能包含舞台、固定道具、相机、灯光或多个不相关 articulated bodies。
- repo assets: 现有资产查找和登记。

`code_agent/asset_bridge` 只做：

- 规范化 planner 的 asset requests。
- 对 mesh 请求调用迁移后的 [mesh](docs/mesh.md) pipeline。
- 对 MJCF/XML 请求调用 Codex XML worker，而不是旧 XML agent。
- XML worker 写完并静态检查后，必须请求一次简单 MuJoCo import validation，确认 XML 语法和模型构建有效。
- XML worker 负责 actuator 设计，并必须在 Asset Manifest 中暴露 actuator/control interface，供 Action Worker 使用。
- 查找或登记 repo 内已有资产。
- 汇总 bbox、scale、runtime path、texture path 和 validation status。
- 将失败压缩成 planner/debugger 可消费的错误摘要。

## 5. Workspace Layout

每个 run 固定使用：

```text
code_agent/workspaces/<run_id>/
  inputs/
    user_prompt.md
    repo_rules.md
    capabilities.md
  contracts/
    planner_output.json
    scene_brief.json
    scene_plan.json
    module_contracts.json
    asset_requests.json
  assets/
    asset_manifest.json
  src/
    scene.py
    body.py
    action.py
    main.py
  logs/
    codex_planner.jsonl
    codex_scene.jsonl
    codex_body.jsonl
    codex_action.jsonl
    codex_integrator.jsonl
    codex_review.jsonl
    execution.stdout
    execution.stderr
  artifacts/
    run_result.json
    event_log.json
    metrics.json
    render.mp4
    frames/
  reports/
    static_review.json
    execution_report.json
    critic.json
    patch_plan.json
  summary.json
```

## 6. Required Schemas

MVP schemas:

- `planner_output.schema.json`
- `asset_manifest.schema.json`
- `worker_report.schema.json`
- `review_report.schema.json`
- `execution_report.schema.json`
- `patch_plan.schema.json`
- `critic_report.schema.json`

这些 schema 只约束流程 metadata，不承担仿真语义编译职责。

## 7. Verification Loop

每轮生成必须通过：

1. Schema validation。
2. Write-scope check。
3. Codex static review。
4. Deterministic policy check：禁止 host Python / `uv` / `pytest` / simulation。
5. Apptainer smoke execution 或 sbatch execution。
6. Artifact completeness check。
7. Task metric check。
8. Video / event critic。

成功准入：

- 代码运行无异常。
- 必要 artifact 完整。
- task metric 通过或有明确合理解释。
- 视频能观察到目标交互。
- critic 没有高严重度物理或语义问题。
- Codex workers 没有越界修改。
- 执行日志没有违反仓库执行规则。

失败路由：

| Failure | Routed To |
| --- | --- |
| syntax/import error | Integrator |
| Genesis API misuse | owning Scene / Body / Action worker plus case library |
| missing asset/path | Asset Bridge or Scene / Body worker |
| fixed object or arena issue | Scene worker |
| movable body placement or geometry issue | Body worker |
| initial overlap | Scene + Body workers |
| solver instability | Scene worker, then Body worker if caused by movable geometry |
| no visible motion | Action worker |
| wrong task semantics | Planner plus Action worker |
| metric/video disagreement | Critic plus Action worker |

Debugger 先输出 `patch_plan.json`，Coordinator 再调用目标 worker 修复。不要默认整场景重写。

## 8. Implementation Phases

### Phase 0: Baseline

- 选 10-20 个 golden tasks，覆盖 rigid、deformable、mesh、MJCF、texture、long-horizon。
- 保存旧 IR 管线输出作为 parity baseline。
- 明确 MVP 必须支持和可延期能力。

### Phase 1: Codex CLI Thin MVP

- 实现 coordinator、workspace manager、Codex invoker、schema validator、Execution Agent。
- 一个 Codex Planner 输出 plan。
- 固定派发 Scene / Body / Action workers，并由 Integrator 生成 runnable project。
- Codex reviewer/debugger 支持第一轮修复。
- 不接入 mesh/MJCF。

成功标准：

- 至少 5 个 primitive 场景自动生成、运行、渲染。
- 常见 syntax/import/API 错误能自动修复一轮。
- `code_agent` 自写代码集中在调用、校验、执行和日志。

### Phase 2: Asset Bridge

- 接入现有 mesh pipeline。
- 接入 MJCF/XML 生成或引用。
- 生成 Asset Manifest。
- Scene / Body workers 只消费 manifest。

### Phase 3: Worker Contract Hardening

- 固化 Scene / Body / Action / Integrator 的 prompt、schema 和 write scope。
- 明确 fixed objects 永远归 Scene；Body 只放参与运动或任务交互的实体。
- Coordinator 校验每个 worker 的 write scope 和 exported symbols。

### Phase 4: Critic and Repair

- 建立 metric + event + video critic。
- Debugger 输出 owner-routed patch plan。
- 支持多轮最小修改。

### Phase 5: Suite and Cost Tracking

- 批量运行 golden tasks。
- 记录每个 Codex invocation 的耗时、token、失败类型和资源使用。
- 比较新 code-native 管线和旧 IR 管线。

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Codex worker 修改越界 | Module Contract + changed-file validation |
| 顶层 planner 递归失控 | Planner 只输出 dispatch plan，Coordinator 负责派发 |
| host-side Python 违规 | worker prompt 禁止，Execution Agent 统一执行，日志审计 |
| Genesis API 幻觉 | helper API、case library、review、execution feedback |
| 删除 IR 后缺少结构化检查 | 保留非执行型 contracts 和 schemas |
| 资产尺度/路径不一致 | Asset Manifest 强制 bbox、scale、runtime path |
| 物理不稳定 | smoke tests、Scene worker、failure cookbook |
| critic 误判 | metrics + event log + video 多证据 |
| Codex CLI 行为变化 | 在 `codex/` 中集中封装 Codex invocation，记录 `codex --version` |

## 10. Relationship To Existing `agent/`

短期：

- `code_agent/` 是新实验管线。
- Codex CLI 是 planner / worker / reviewer / debugger backend。
- [mesh](docs/mesh.md) 作为 asset service 复用；MJCF/XML 由 Codex XML worker 生成或从 repo 资产引用。
- `agent/docs` 继续描述旧 IR 管线。

中期：

- 新旧管线共享 mesh/texture 资产层和 suite summaries。
- 将成功 prompts、schemas、case snippets 沉淀到 `code_agent/case_library`。

长期：

- 新 code-native 管线成为默认自然语言入口。
- 旧 IR 管线保留为 legacy baseline 或结构化输入 fallback。

## 11. References

设计参考仅保留方向性依据：

- MCP-SIM: multi-agent physics simulation self-correction.
- GenSim / GenSim2: LLM-generated robotic simulation tasks and task libraries.
- RoboGen: propose-generate-learn loop for embodied simulation data.
- Eureka / Code as Policies: executable code generation with feedback.
- AgentCoder / MetaGPT / ChatDev / AutoGen / SWE-agent: multi-agent software engineering patterns.
