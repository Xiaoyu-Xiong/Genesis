# Generation and Optimization

This document covers IR generation, multimodal critique, and the iterative optimization loop.

## Generator

[agent/llm_generator/cli.py](../llm_generator/cli.py) generates IR from natural language.

Current generator flow can trigger:

- articulated XML generation
- non-articulated mesh generation

Texture generation for mesh assets can be enabled explicitly with:

- `--mesh-texture-enabled`

Example:

```bash
uv run python -m agent.llm_generator.cli generate \
  --task "Create a scene with soft mesh props and a rigid pusher." \
  --model gpt-5.4 \
  --reasoning-effort high \
  --mesh-texture-enabled \
  --out /tmp/generated_ir.json \
  --log-out /tmp/generation.log.json
```

### Tool-Library Policy

[agent/tool_library](../tool_library) contains:

- generator-side hard rules
- mesh / articulated tool specifications
- validation logic
- program constraints

This layer is responsible for:

- keeping prompts within current runtime capability
- routing articulated XML generation
- routing mesh generation
- enforcing constraints such as density bounds and initial-scene validation
- sanitizing generator payloads before schema parse, including stripping deformable-body collision fields that are unsupported in the current FEM+IPC pipeline

## Critic

[agent/llm_critic/cli.py](../llm_critic/cli.py) evaluates:

- task
- IR
- optional XML
- event pack
- rendered video

Output includes:

- `verdict`
- `overall_score`
- `summary`
- `by_section`
- `by_body`
- `priority_fixes`

## Optimization Loop

[agent/opt/cli.py](../opt/cli.py) provides:

- `optimize`
- `optimize-batch`

Optimization loop:

1. generate IR
2. validate and normalize
3. execute simulation
4. build event pack and render video
5. critique with multimodal critic
6. feed structured feedback into the next round

The optimization CLI also supports:

- `--mesh-texture-enabled`

so the full loop can request textured mesh assets when needed.

### Typical Optimize Command

```bash
uv run python -m agent.opt.cli optimize \
  --task "Create a contact-rich deformable scene with soft mesh props." \
  --out-dir agent/runs/example_opt \
  --out agent/runs/example_opt/summary.json \
  --mesh-texture-enabled
```

## Central Config

[agent/configs.py](../configs.py) is the central static configuration module.

It currently contains:

- `RuntimeConfigs`
- `DeformableConfigs`
- `OptimizationConfigs`
- `MeshyRequestConfigs`
- `MeshRepairConfigs`

Notable rules:

- config values are static Python defaults
- they are no longer loaded from environment variables
- run-specific behavior should prefer explicit CLI flags
