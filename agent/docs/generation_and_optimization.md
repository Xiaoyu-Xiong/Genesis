# Generation and Optimization

This document covers IR generation, multimodal critique, and the iterative optimization loop.

## Generator

[agent/llm_generator/cli.py](../llm_generator/cli.py) generates IR from natural language.

Current generator flow can trigger:

- articulated XML generation
- non-articulated mesh generation

Current OpenAI-facing pipeline details:

- generator requests use Responses API state with `previous_response_id`
- generator and critic requests now default to `prompt_cache_retention="24h"`
- batch runs emit per-round OpenAI usage artifacts via `llm_usage.json`
- `optimize-batch` supports `--max-parallel` for memory-heavy suites
- for generated mesh bodies, the main runtime IR must use the canonical repaired runtime mesh path returned as `mesh_path` (typically `processed/repaired*.obj`), not auxiliary textured OBJ assets
- during active runs, `generation.log.json` and `critic.log.json` are refreshed incrementally as generator / critic tool rounds complete, so partially populated logs during execution are expected
- critic video sampling is now unified on OpenCV for both duration probing and frame extraction; when downscaling is needed, frames are resized with OpenCV Lanczos4

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

[agent/tool_library](../tool_library) is the policy layer between prompt-side generation and runtime capability.

Current structure is:

- payload and tool-bootstrap code:
  - [payloads/generation.py](../tool_library/payloads/generation.py)
  - [payloads/specs.py](../tool_library/payloads/specs.py)
  - `tool_library/payloads/`
- constraint and rule code:
  - [constraints/program.py](../tool_library/constraints/program.py)
  - [constraints/rules.py](../tool_library/constraints/rules.py)
  - `tool_library/constraints/`
- runtime adapters:
  - [runtime_api.py](../tool_library/runtime_api.py)
  - [capabilities.py](../tool_library/capabilities.py)
  - [overrides.py](../tool_library/overrides.py)

This layer is responsible for:

- keeping prompts within current runtime capability
- routing articulated XML generation
- routing mesh generation
- enforcing constraints such as density bounds and initial-scene validation
- sanitizing generator payloads before schema parse, including stripping deformable-body collision fields that are unsupported in the current FEM+IPC pipeline and rewriting mistakenly selected generated textured OBJ paths back to the canonical repaired runtime mesh path when available

For the current `agent` FEM+IPC runtime path, all IR rigid bodies, including fixed obstacles and articulated rigid
bodies, are routed onto IPC `two_way_soft_constraint` coupling in both direct runtime execution and compiled output.
This keeps soft-rigid contact available through IPC while keeping the rigid bodies visible to Genesis' rigid solver
for pure rigid contact whenever `deformable.ipc_enable_rigid_rigid_contact=false`. The hidden FEM support ground remains
an IPC-only plane; normal `scene.add_ground` still adds a Genesis rigid ground for rigid bodies.

For FEM+IPC sanity failures, the validator now preserves more of libuipc's original diagnostic lines, including `SimplicialSurfaceIntersectionCheck` output, and annotates runtime object names such as `fem_0_0` or `rigid_link_3_0` back to inferred IR body names when that mapping can be recovered before scene build. This feedback is returned through `validate_ir` tool errors and through the generator's local finalization error path, so revision rounds can see specific overlapping body pairs instead of only a generic "world is not valid" message. When a specific geometric root cause such as initial intersection or half-plane clearance failure is already present, the validator suppresses later cascading `rigid state accessor` noise from the same failed build so the generator is not steered by the wrong secondary error.

Observation contact lists in the runtime/event-pack are currently unified onto AABB-overlap heuristics for both rigid and deformable bodies. Critic prompting explicitly treats these lists as weak spatial-overlap signals only; they are neither sufficient nor necessary evidence of true physical contact and must be cross-checked against video, motion, and deformation evidence.

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
- `--max-parallel` override for memory-heavy batch runs

Optimization loop:

1. generate IR
2. validate and normalize
3. execute simulation
4. build event pack and render video
5. critique with staged multimodal critic
6. feed structured feedback into the next round

The optimization CLI also supports:

- `--mesh-texture-enabled`

so the full loop can request textured mesh assets when needed.

The internal optimization implementation is split into:

- [pipeline.py](../opt/pipeline.py): round orchestration and batch control
- [models.py](../opt/models.py): config and result dataclasses
- [artifacts.py](../opt/artifacts.py): workspace layout, run payload shaping, and usage/artifact helpers

The current critic path is two-stage when enabled in [agent/configs.py](../configs.py):

1. stage 1 compact screening
2. stage 2 retrieval-based critic only when stage 1 escalates

By default, a stage-1 `pass` still forces escalation into stage 2 for a second check. This is meant to catch false-positive compact passes in visually ambiguous cases.

Usage accounting is preserved per component:

- generator IR
- generator XML
- critic stage 1
- critic stage 2

Suite runs can be summarized with:

- [agent/scripts/summarize_openai_usage.py](../scripts/summarize_openai_usage.py)

This produces run-root summaries such as:

- `openai_usage_summary.json`
- `openai_usage_summary.tsv`

For memory-heavy texture suites, prefer lowering `optimize-batch --max-parallel` instead of letting all cases launch at the default worker count. Process-pool failures with empty round lists usually indicate a worker was killed externally (for example by OOM) before the normal per-case error handling could run.

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

### Runtime And IPC Knobs

For FEM+IPC scenes, the most important stability knobs now live in
[agent/configs.py](../configs.py) under `RuntimeConfigs` and `DeformableConfigs`.

- `runtime.sim_dt`: top-level simulation step duration in seconds. Smaller values reduce integration error and
  contact jitter at higher runtime cost.
- `runtime.sim_substeps`: number of internal solver substeps per `scene.step()`. Raising this improves contact-rich
  stability without changing the scene-level duration encoded by IR step counts, but increases runtime roughly
  proportionally.
- `deformable.ipc_newton_max_iterations`: cap on IPC Newton iterations per step. Higher values can improve
  convergence in hard contact scenes, at extra cost.
- `deformable.ipc_newton_min_iterations`: minimum Newton iterations before early termination is allowed. Useful when
  the solver exits too aggressively on noisy scenes.
- `deformable.ipc_newton_tolerance`: velocity convergence tolerance for Newton. Smaller values are stricter and
  usually more stable, but slower.
- `deformable.ipc_newton_ccd_tolerance`: CCD tolerance for Newton. Smaller values make collision handling stricter
  and can reduce tunneling or late collision response.
- `deformable.ipc_newton_use_adaptive_tolerance`: whether Newton tolerance adapts during solve. This can improve
  robustness when scenes range from easy to stiff contact.
- `deformable.ipc_newton_translation_tolerance`: translation-rate convergence threshold. Lower values force tighter
  convergence on residual rigid/soft motion.
- `deformable.ipc_newton_semi_implicit_enable`: enables libuipc semi-implicit Newton mode. This is often worth
  testing first when free rigid bodies coupled to soft bodies show persistent oscillation.
- `deformable.ipc_newton_semi_implicit_beta_tolerance`: auxiliary tolerance for semi-implicit Newton mode. Tighten
  only after deciding to use semi-implicit solve at all.
- `deformable.ipc_n_linesearch_iterations`: maximum line-search backtracking iterations. Higher values can make
  difficult contact solves more robust when the first Newton step overshoots.
- `deformable.ipc_linesearch_report_energy`: debug flag that asks libuipc to report line-search energy.
- `deformable.ipc_linear_system_solver`: inner linear solver choice. `linear_pcg` is usually cheaper; `direct` may be
  more robust on small but stiff scenes if supported by the local libuipc build.
- `deformable.ipc_linear_system_tolerance`: tolerance for the linear solver. Smaller values mean a more accurate but
  slower inner solve.
- `deformable.ipc_contact_enable`: global contact on/off switch for IPC. Mostly useful for debugging rather than
  tuning production scenes.
- `deformable.ipc_contact_d_hat`: contact activation distance. Larger values detect contact earlier and can soften
  sharp impacts; too large can make contacts look mushy.
- `deformable.ipc_contact_friction_enable`: enables IPC friction.
- `deformable.ipc_contact_resistance`: default contact resistance / stiffness fallback used when a material does not
  define its own IPC resistance. Larger values make contact harder and can increase chatter if other settings are too
  aggressive.
- `deformable.ipc_contact_eps_velocity`: low-speed friction regularization threshold. Increasing it often helps
  suppress stick-slip jitter and noisy contact reversals.
- `deformable.ipc_contact_constitution`: contact law variant such as `ipc` or `isometric`.
- `deformable.ipc_collision_detection_method`: collision-detection backend.
- `deformable.ipc_cfl_enable`: enables libuipc CFL safeguards. This can improve robustness in aggressive scenes that
  otherwise take too-large effective updates.
- `deformable.ipc_sanity_check_enable`: enables libuipc sanity checks, which are useful while debugging scene setup
  and solver pathologies.
- `deformable.ipc_constraint_strength_translation`: strength of soft transform constraints that couple Genesis rigid
  bodies to IPC rigid representations. Higher values make the coupling stiffer and can amplify high-frequency
  feedback.
- `deformable.ipc_constraint_strength_rotation`: rotational counterpart of the above. Lowering this is often useful
  when a free rigid plate jitters in roll/pitch while pressing soft bodies.
- `deformable.ipc_enable_rigid_ground_contact`: whether rigid-ground pairs also participate in IPC.
- `deformable.ipc_enable_rigid_rigid_contact`: whether rigid-rigid pairs participate in IPC. When this is false, the
  current agent rigid-body routing leaves pure rigid-rigid contact on Genesis' rigid solver while FEM-involving contact
  remains in IPC.
- `deformable.ipc_two_way_coupling`: whether IPC reaction forces feed back into Genesis rigid bodies. Disabling this
  can reduce jitter, but it also removes physically important soft-to-rigid feedback.
- `deformable.ipc_enable_rigid_dofs_sync`: whether IPC reference DOF state is synchronized from Genesis each step.
  Tightening this can help some articulated cases but may amplify small state divergence.
- `deformable.ipc_free_base_driven_by_ipc`: whether a free-base rigid body is fully driven by IPC rather than by the
  Genesis-side soft transform constraint. This is a high-impact switch for scenes with free rigid plates or pushers
  interacting with soft bodies.
