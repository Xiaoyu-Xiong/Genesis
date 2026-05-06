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
