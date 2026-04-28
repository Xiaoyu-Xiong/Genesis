# Agent Ownership and Collaboration

This document defines the current subagent-style ownership boundaries for the `agent/` pipeline.

The goal is not to create hard organizational walls. The goal is to make edits easier to route, reduce ownership ambiguity,
and make cross-cutting reviews more predictable.

## Ownership Model

Each area should have:

- one primary owner
- zero or more required reviewers for shared boundary files

When a file is listed under "shared boundary files", the primary owner drives implementation, but the listed reviewers
should review changes that affect their concern area.

## Subagents

### 1. Physics Runtime Agent

Scope:

- IR semantics
- runtime construction
- physical parameter meaning
- execution results and event-pack semantics
- rigid / deformable / FEM / IPC / PBD behavior at the system level

Primary ownership:

- `agent/ir_schema/**`
- `agent/runtime/**`
- `agent/compiler_backend/**`
- `agent/configs.py`
- `agent/cli.py`

Key files:

- [body.py](../ir_schema/body.py)
- [actions.py](../ir_schema/actions.py)
- [builders.py](../runtime/builders.py)
- [helpers.py](../runtime/helpers.py)
- `runtime/observation/`
- [setup.py](../runtime/setup.py)
- [runner.py](../runtime/runner.py)
- [event_pack.py](../runtime/event_pack.py)

Shared boundary files it must review:

- [program.py](../tool_library/constraints/program.py)
- [rules.py](../tool_library/constraints/rules.py)
- [genesis/utils/mesh.py](../../genesis/utils/mesh.py)
- [genesis/utils/element.py](../../genesis/utils/element.py)
- [fem_entity.py](../../genesis/engine/entities/fem_entity.py)

Typical tasks:

- changing `simulation_kind`, `fixed`, `rho`, `E`, `nu`, or collision semantics
- changing runtime builders or execution flow
- changing event-pack meaning or schema-level behavior
- changing deformable / rigid coupling behavior

Out of scope:

- Meshy request logic
- texture transfer algorithms
- generator / critic wording
- batch suite orchestration

### 2. Mesh Agent

Scope:

- mesh asset lifecycle
- Meshy generation
- repair and manifold readiness
- texture refine, transfer, and textured validation render
- mesh / texture integration into the deformable render path

Primary ownership:

- `agent/mesh/**`
- `agent/llm_generator/agents/mesh_agent.py`

Key files:

- [pipeline.py](../mesh/pipeline.py)
- [workflow/steps.py](../mesh/workflow/steps.py)
- [workflow/meshy.py](../mesh/workflow/meshy.py)
- [workflow/summary.py](../mesh/workflow/summary.py)
- [repair/postprocess.py](../mesh/repair/postprocess.py)
- [repair/backend.py](../mesh/repair/backend.py)
- [repair/sanity.py](../mesh/repair/sanity.py)
- [texture/transfer.py](../mesh/texture/transfer.py)
- [texture/render_views.py](../mesh/texture/render_views.py)

Shared boundary files it drives, with required Physics Runtime review:

- [genesis/utils/mesh.py](../../genesis/utils/mesh.py)
- [genesis/utils/element.py](../../genesis/utils/element.py)
- [fem_entity.py](../../genesis/engine/entities/fem_entity.py)

Typical tasks:

- Meshy failures
- repair failures
- TetGen self-intersection issues
- mesh `scale`, bbox, manifold, and texture-transfer issues
- deformable render artifact construction for textured meshes

Out of scope:

- IR action semantics
- critic scoring logic
- optimization feedback strategy
- batch suite prompt design

### 3. Prompt Policy Agent

Scope:

- generator and critic policy
- tool-library behavior
- program constraints
- prompt-side parameter guidance
- optimization feedback and revision strategy

Primary ownership:

- `agent/tool_library/**`
- `agent/llm_generator/constraints/**`
- `agent/llm_critic/**`
- `agent/opt/**`

Key files:

- [ir_agent.py](../llm_generator/agents/ir_agent.py)
- [two_agent_generator.py](../llm_generator/agents/two_agent_generator.py)
- [xml_agent.py](../llm_generator/agents/xml_agent.py)
- [prompt_utils.py](../llm_generator/agents/prompt_utils.py)
- [constraints/rules.py](../tool_library/constraints/rules.py)
- [constraints/program.py](../tool_library/constraints/program.py)
- [payloads/generation.py](../tool_library/payloads/generation.py)
- [payloads/specs.py](../tool_library/payloads/specs.py)
- [prompting.py](../llm_critic/prompting.py)
- [pipeline.py](../opt/pipeline.py)
- [models.py](../opt/models.py)
- [artifacts.py](../opt/artifacts.py)
- [feedback.py](../opt/feedback.py)

Shared boundary files requiring specialist review:

- [body.py](../ir_schema/body.py) with Physics Runtime review
- [builders.py](../runtime/builders.py) with Physics Runtime review
- [mesh_agent.py](../llm_generator/agents/mesh_agent.py) with Mesh Agent review

Typical tasks:

- generator / critic prompt changes
- tool schema and tool-rule changes
- parameter recommendations such as `rho`, `E`, `nu`, `scale`
- penetration fallback policy
- mesh reuse and scene-design prompting policy

Out of scope:

- low-level mesh repair or texture-transfer implementation
- runtime builder internals
- suite shell orchestration

### 4. Suite Ops Agent

Scope:

- suite scripts
- run orchestration
- summary and artifact organization
- agent-facing docs and usage guides

Primary ownership:

- `agent/scripts/**`
- `agent/README.md`
- `agent/docs/**`

Key files:

- [run_opt_deformable_texture_suite.sh](../scripts/run_opt_deformable_texture_suite.sh)
- [run_opt_deformable_mesh_suite.sh](../scripts/run_opt_deformable_mesh_suite.sh)
- [run_mesh_meshy_texture_suite.sh](../scripts/run_mesh_meshy_texture_suite.sh)
- [README.md](../README.md)
- [suites_and_artifacts.md](suites_and_artifacts.md)

Shared boundary CLIs it should coordinate around:

- [agent/opt/cli.py](../opt/cli.py)
- [agent/llm_generator/cli.py](../llm_generator/cli.py)
- [agent/mesh/cli.py](../mesh/cli.py)

Typical tasks:

- adding or updating benchmark suites
- standardizing output roots, summaries, and failure logs
- documenting usage patterns and run conventions
- preparing reproducible `sbatch` or direct-run commands

Out of scope:

- physical semantics
- prompt policy internals
- mesh-repair algorithms
- runtime builder implementation

## Shared Boundary Files

These files cross ownership boundaries and should be treated explicitly.

### `genesis/utils/mesh.py`

- primary owner: Mesh Agent
- required reviewer: Physics Runtime Agent

Reason:

- it contains both mesh preprocessing logic and behavior that materially affects tet / FEM semantics

### `genesis/utils/element.py`

- primary owner: Mesh Agent
- required reviewer: Physics Runtime Agent

Reason:

- it is the junction between mesh preprocessing, tetrahedralization, and render-facing artifact construction

### `genesis/engine/entities/fem_entity.py`

- runtime semantics owner: Physics Runtime Agent
- texture / render-path owner: Mesh Agent

Reason:

- it mixes FEM runtime behavior with render-facing surface and UV handling

### `agent/llm_generator/agents/two_agent_generator.py`

- primary owner: Prompt Policy Agent
- required reviewer: Mesh Agent

Reason:

- it decides when mesh generation is invoked and how generated mesh assets enter the IR flow

### `agent/tool_library/constraints/program.py`

- primary owner: Prompt Policy Agent
- required reviewers:
  - Physics Runtime Agent
  - Mesh Agent

Reason:

- it is the policy boundary where runtime capability and mesh stability constraints are enforced

### `agent/opt/pipeline.py`

- primary owner: Prompt Policy Agent
- required reviewer: Suite Ops Agent

Reason:

- it defines optimization-loop behavior, while suite scripts directly consume its run outputs

## Collaboration Contract

These rules keep the four areas aligned.

1. Physics Runtime Agent defines the actual capability boundary.
   Prompts, suites, and mesh-side assumptions must not claim behavior the runtime does not implement.

2. Mesh Agent defines the mesh asset contract.
   This includes mesh directory structure, metadata shape, repaired outputs, and textured asset expectations.

3. Prompt Policy Agent builds only on published runtime and mesh capabilities.
   It should not imply hidden features or unsupported repair / render behavior.

4. Suite Ops Agent should drive public CLIs, not internal helper functions.
   Suites should validate real user entrypoints instead of depending on private module internals.

5. Shared boundary files should not be treated as "free for all" edit zones.
   They require ownership-aware review whenever semantics cross from one area into another.

## When to Create a New Ownership Boundary

Consider creating a separate ownership slice only if all three are true:

- the new area has a stable technical boundary
- it has repeated tasks that are materially different from the current owner groups
- splitting it reduces review ambiguity instead of increasing it

At the current repo state, four ownership areas are enough. A fifth one would likely create more coordination cost than value.
