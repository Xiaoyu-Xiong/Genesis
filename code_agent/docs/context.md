# Genesis Context

`context/genesis.py` builds a compact context pack once per suite and copies pointers into each case's `contracts/`
directory.

The pack covers the current code-agent scope:

- ordinary rigid scenes
- rigid/articulated scenes with IPC contact/coupling
- FEM deformables with IPC
- FEM.Cloth thin-shell cloth with IPC
- generated meshes and XML/MJCF assets
- rendering, visual evidence, execution, and critique

Out-of-scope solver families such as MPM, PBD, SPH, and fluid are filtered out of generated context unless they are
explicitly reintroduced later. PBD cloth remains out of scope; cloth tasks should use FEM.Cloth with IPC.

Files written under the suite root:

```text
context/genesis/genesis_context.md
context/genesis/genesis_context.json
context/genesis/official_catalog.json
context/genesis/official_docs/
```

Agents receive paths to these files and should read only the relevant source/docs. Local Genesis source and examples
win over online docs when they disagree.

## Human Experience Debug Cards

The simdebug context layer lives beside the Genesis context pack:

```text
code_agent/context/simdebug/
  catalog.json
  assets_geometry/
  contact_collision/
  control_dynamics/
  deformable_fem/
  diagnosis_repair/
  evidence_validation/
  optimization/
  workflow_orchestration/
```

Do not add a separate `README.md` under `context/simdebug/`. This file is the documentation entry point for context
assets.

Simdebug cards store prompt-derived and human-authored simulation debugging experience as structured YAML cards. The
card library uses one unified card type: each card may contain `guidance`, `restrictions`, `checks`, and
`dispatch_hints` as needed instead of being classified as a guideline card or restriction card.

Planner owns card selection and dispatch: it reads the catalog, selects all cards it judges relevant from the global
case state, and passes summarized card bundles to writers, Critic, and Opt. Downstream agents should not read the card
library directly.

Cards are organized in direct subdirectories under `context/simdebug/` by simulation problem domain rather than by
agent owner or guidance/restriction kind:

- `assets_geometry`: generated meshes, XML/MJCF assets, shape fidelity, texture requests, asset manifests, and asset inspection.
- `contact_collision`: collision/contact enablement, IPC coupling, scale, initial clearance, dummy mounts, and runtime contact config.
- `control_dynamics`: staged controllers, force/gain limits, phase gates, release/retreat behavior, and physical causality.
- `deformable_fem`: FEM material, soft-body layout, FEM state metrics, deformable scope, and FEM actuation channels.
- `diagnosis_repair`: IPC failure signatures, invalid-world diagnosis, source-aware repair, and repair routing.
- `evidence_validation`: visual evidence, render clarity, metrics, and fresh-artifact consistency checks.
- `optimization`: Opt routing, search space, objectives, effective parameters, task semantics, and XML scalar patches.
- `workflow_orchestration`: Planner-owned card dispatch and other cross-cutting orchestration rules.

The directory name is a maintenance hint for the problem domain. The `scopes` field inside each card remains the source
of truth for every role that Planner may dispatch the card to.

The first card set should migrate every `code_agent/prompts/` clause that is better represented as structured debugging
experience. Prompts should keep general role, protocol, schema, and edit-boundary instructions; situational debugging
knowledge should move into cards so it can be selected, audited, ablated, and revised independently.
