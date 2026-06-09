# Genesis Context

`context/genesis.py` builds a compact context pack once per suite and copies pointers into each case's `contracts/`
directory.

The pack covers the current code-agent scope:

- ordinary rigid scenes
- rigid/articulated scenes with IPC contact/coupling
- FEM deformables with IPC
- generated meshes and XML/MJCF assets
- rendering, visual evidence, execution, and critique

Out-of-scope solver families such as MPM, PBD, SPH, fluid, and cloth are filtered out of generated context unless they
are explicitly reintroduced later.

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
  cards/
    planner/
    scene/
    body/
    action/
    rendering/
    critic/
    opt/
```

Do not add a separate `README.md` under `context/simdebug/`. This file is the documentation entry point for context
assets.

Simdebug cards store prompt-derived and human-authored simulation debugging experience as structured YAML cards. The
card library uses one unified card type: each card may contain `guidance`, `restrictions`, `checks`, and
`dispatch_hints` as needed instead of being classified as a guideline card or restriction card.

Planner owns card selection and dispatch: it reads the catalog, selects all cards it judges relevant from the global
case state, and passes summarized card bundles to writers, Critic, and Opt. Downstream agents should not read the card
library directly.

Cards under `cards/` are organized by the primary agent owner rather than by topic or guidance/restriction kind:

- `planner`: planning, dispatch, asset-request, and high-level routing cards.
- `scene`: layout, scale, initial geometry, and scene-configuration cards.
- `body`: entity construction, assets, material, collision, XML/MJCF, and coupling cards.
- `action`: controller, runtime evidence, metric, and physical-causality cards.
- `rendering`: camera, lighting, background, and visual-evidence cards.
- `critic`: failure-classification, asset-evaluation, and source-aware repair cards.
- `opt`: Opt routing, parameter-space, objective, semantic, and XML patch cards.

The directory name is the card's primary owner for maintainability. The `scopes` field inside each card remains the
source of truth for every role that Planner may dispatch the card to.

The first card set should migrate every `code_agent/prompts/` clause that is better represented as structured debugging
experience. Prompts should keep general role, protocol, schema, and edit-boundary instructions; situational debugging
knowledge should move into cards so it can be selected, audited, ablated, and revised independently.
