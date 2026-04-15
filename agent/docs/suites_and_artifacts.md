# Suite Scripts and Artifacts

This document summarizes the main batch scripts and the artifact roots they produce.

## Suite Scripts

Useful scripts under [agent/scripts](../scripts):

- [run_opt_robot_suite.sh](../scripts/run_opt_robot_suite.sh)
- [run_opt_multibody_suite.sh](../scripts/run_opt_multibody_suite.sh)
- [run_opt_mesh_suite.sh](../scripts/run_opt_mesh_suite.sh)
- [run_opt_mesh_realworld_suite.sh](../scripts/run_opt_mesh_realworld_suite.sh)
- [run_opt_deformable_suite.sh](../scripts/run_opt_deformable_suite.sh)
- [run_opt_deformable_mesh_suite.sh](../scripts/run_opt_deformable_mesh_suite.sh)
- [run_opt_deformable_texture_suite.sh](../scripts/run_opt_deformable_texture_suite.sh)
- [run_opt_multiarticulated_suite.sh](../scripts/run_opt_multiarticulated_suite.sh)
- [run_opt_multibody_compare_suite.sh](../scripts/run_opt_multibody_compare_suite.sh)
- [run_mesh_meshy_suite.sh](../scripts/run_mesh_meshy_suite.sh)
- [run_mesh_meshy_texture_suite.sh](../scripts/run_mesh_meshy_texture_suite.sh)

Texture-specific scripts are split by intent:

- `run_mesh_meshy_texture_suite.sh`: standalone textured mesh generation / repair / render validation
- `run_opt_deformable_texture_suite.sh`: full task optimization with deformable textured bodies

## Generated Artifact Roots

Current artifact roots:

- `agent/generated_assets/`: generated articulated XML assets grouped by task/run
- `agent/generated_meshes/`: generated mesh assets grouped by task/run
- `agent/runs/`: optimization runs, videos, summaries, and suite outputs

Common run roots:

- `agent/runs/opt_robot_suite`
- `agent/runs/opt_multibody_suite`
- `agent/runs/opt_mesh_suite`
- `agent/runs/opt_mesh_realworld_suite`
- `agent/runs/opt_deformable_suite`
- `agent/runs/opt_deformable_mesh_suite`
- `agent/runs/opt_deformable_texture_suite`
- `agent/runs/mesh_meshy_texture_suite`

## Practical Workflow

### Hand-authored IR

1. write IR JSON
2. validate
3. run
4. inspect `event_pack.json`
5. optionally compile generated Genesis Python

### Natural-language generation

1. generate with `agent.llm_generator.cli`
2. validate and run
3. inspect `run_result.json`, `event_pack.json`, and `render.mp4`
4. critique with `agent.llm_critic.cli`
5. use `agent.opt.cli` when iterative repair is needed

### Standalone textured mesh validation

1. generate mesh with `agent.mesh.cli generate --generate-texture`
2. inspect `raw_manifold_check.json` and `manifold_check.json`
3. inspect `processed/repaired.obj`, `processed/repaired.mtl`, and `processed/base_color.png`
4. render multi-view PNGs with `agent.mesh.cli render-textured-views`
5. if needed, inspect debug outputs under `processed/repaired_texture_debug`
