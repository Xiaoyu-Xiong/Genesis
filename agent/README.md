# Agent Pipeline

This directory contains the natural-language-to-Genesis pipeline used in this repo.

Current scope includes:

- rigid scenes
- deformable scenes
- articulated bodies via MJCF / URDF
- non-articulated mesh bodies
- Meshy-based mesh generation
- textured mesh generation, repair, and transfer
- multimodal critique and iterative optimization

The top-level IR class is still named `RigidIR` for historical reasons, but the pipeline is no longer rigid-only.

## Execution Rules

Repository-wide execution rules are defined in [AGENTS.md](../AGENTS.md).

The practical summary for `agent/` work is:

- run Python through the repository uv environment
- run git from the repository shell
- prefer explicit CLI flags over ad hoc config mutation

## Pipeline-Specific Maintenance Rules

These rules apply to the current legacy `agent/` pipeline. They are intentionally scoped here so the repository root can
stay focused on rules that apply across both the existing IR-based pipeline and future code-native work.

- `agent/configs.py` is a static config module. Do not reintroduce environment-variable-driven config loading there.
  For run-specific behavior, prefer explicit CLI flags.
- When changing the `agent/` pipeline, suite scripts, or mesh / texture / deformable workflows, update the relevant
  documentation under `agent/` in the same turn.
- For `agent/` work, if the user does not explicitly prescribe a task split, decide whether to decompose the work across
  the ownership areas documented in [Ownership and Collaboration](docs/ownership.md), and choose the split that minimizes
  ambiguity and coordination cost. Explicit user instructions about scope, ownership, or task splitting always override
  this default.

## Documentation Map

- [IR and Runtime](docs/ir_runtime.md)
- [Generation and Optimization](docs/generation_and_optimization.md)
- [Mesh and Texture Pipeline](docs/mesh_texture.md)
- [Agentive-Native Upgrade Plan](docs/agentive_native_pipeline.md)
- [Suite Scripts and Artifacts](docs/suites_and_artifacts.md)
- [Ownership and Collaboration](docs/ownership.md)

## Ownership Summary

The current `agent/` pipeline is organized around four ownership areas:

- `Physics Runtime Agent`: owns IR semantics, runtime construction, execution behavior, and event-pack meaning.
- `Mesh Agent`: owns Meshy generation, repair, manifold readiness, texture transfer, and mesh/texture integration into deformable rendering.
- `Prompt Policy Agent`: owns generator / critic policy, tool-library rules, program constraints, and optimization feedback strategy.
- `Suite Ops Agent`: owns benchmark scripts, run orchestration, artifact summaries, and `agent/`-side documentation.

Important shared-boundary files are not "free edit" zones. In particular:

- `genesis/utils/mesh.py`
- `genesis/utils/element.py`
- `genesis/engine/entities/fem_entity.py`
- `agent/llm_generator/agents/two_agent_generator.py`
- `agent/tool_library/constraints/program.py`
- `agent/opt/pipeline.py`

For the detailed ownership matrix, shared-file review expectations, and collaboration contract, see
[Ownership and Collaboration](docs/ownership.md).

## Main Entry Points

- [agent/cli.py](cli.py): validate, compile, and run IR
- [agent/llm_generator/cli.py](llm_generator/cli.py): generate IR from text
- [agent/llm_critic/cli.py](llm_critic/cli.py): critique task + IR + video
- [agent/opt/cli.py](opt/cli.py): iterative optimize / optimize-batch
- [agent/mesh/cli.py](mesh/cli.py): standalone mesh generation, manifold check, and textured renders
- [agent/configs.py](configs.py): static central config

## Main Subdirectories

- `ir_schema/`
- `runtime/`
- `runtime/observation/`
- `compiler_backend/`
- `tool_library/`
- `tool_library/payloads/`
- `tool_library/constraints/`
- `llm_generator/`
- `llm_critic/`
- `mesh/`
- `mesh/repair/`
- `mesh/texture/`
- `mesh/workflow/`
- `opt/`
- `scripts/`
- `generated_assets/`
- `generated_meshes/`
- `runs/`

## Common Workflow

### Hand-authored IR

1. validate with `agent.cli validate`
2. run with `agent.cli run`
3. inspect `run_result.json` and `event_pack.json`
4. optionally compile with `agent.cli compile`

### Natural-language optimization

1. generate with `agent.llm_generator.cli`
2. or run the full loop with `agent.opt.cli optimize` / `optimize-batch`
3. inspect `generation.log.json`, `critic.json`, `run_result.json`, and `render.mp4`

### Standalone textured mesh validation

1. generate with `agent.mesh.cli generate --generate-texture`
2. inspect `raw_manifold_check.json` and `manifold_check.json`
3. inspect `processed/repaired.obj`, `processed/repaired.mtl`, and `processed/base_color.png`
4. render validation views with `agent.mesh.cli render-textured-views`
