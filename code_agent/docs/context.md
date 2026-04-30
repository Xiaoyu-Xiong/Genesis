# Genesis Context

`code_agent/context/` builds the Genesis context pack that planner, writer, and critic agents receive for each suite
case.

The active scope is intentionally narrow: FEM deformables with IPC coupling. Rigid bodies, articulated MJCF/URDF robots,
generated meshes, textures, and rendering remain in scope only because they are common participants in FEM+IPC scenes.
Other non-rigid families such as MPM, PBD, SPH, drones, terrain generation, and generic sensor work are not part of the
current code-agent target.

## Runtime Behavior

At suite startup, `utils/suite.py` calls `build_genesis_context_pack(out_dir)`. The builder fetches and caches selected
official Genesis documentation pages under:

```text
<suite_out_dir>/context/genesis/official_docs/
```

It also writes:

```text
<suite_out_dir>/context/genesis/genesis_context.md
<suite_out_dir>/context/genesis/genesis_context.json
<suite_out_dir>/context/genesis/official_catalog.json
```

Each case receives copies under `contracts/genesis_context.md` and `contracts/genesis_context.json`. Planner, writer,
and critic prompts include only a short pointer to these files and to the cached docs directory. They are expected to
open the relevant files on demand instead of receiving the full context pack inline.

The selected official pages are filtered to remove out-of-scope solver families before they are cached. This keeps the
agent context focused on FEM+IPC rather than letting MPM/PBD/SPH-oriented examples leak back into generation prompts.

## Agent Use

Planner receives the context pack directly in its prompt and should use it to produce detailed FEM+IPC-aware plans.
Writer agents receive the same pack plus paths to the workspace, contracts, assets, reports, and artifacts. The critic
receives the pack together with generated source and run evidence, so repair feedback can compare the user prompt,
source, official docs, local code anchors, metrics, logs, and visual output.

The cached documentation is not meant to override local source. If official docs and this checkout disagree, local
Genesis source and examples win.
