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
  guideline_cards/
    assets/
    ipc/
    opt/
    repair_routing/
  restriction_cards/
    assets/
    evidence/
    opt/
    physics_validity/
```

Do not add a separate `README.md` under `context/simdebug/`. This file is the documentation entry point for context
assets.

Simdebug cards store prompt-derived and human-authored simulation debugging experience as structured YAML cards.
Planner owns card selection and dispatch: it reads the catalog, selects all cards it judges relevant from the global
case state, and passes summarized card bundles to writers, Critic, and Opt. Downstream agents should not read the card
library directly.

Cards are split into:

- `guideline`: diagnostic and repair guidance, such as FEM material selection, IPC failure diagnosis, Opt variable
  choice, metric design, and repair routing.
- `restriction`: acceptance guards, such as physical-causality checks, collision/contact requirements, visual-evidence
  requirements, and invalid Opt shortcut checks.

The first card set should migrate every `code_agent/prompts/` clause that is better represented as a guideline or
restriction card. Prompts should keep general role, protocol, schema, and edit-boundary instructions; situational
debugging knowledge should move into cards so it can be selected, audited, ablated, and revised independently.
