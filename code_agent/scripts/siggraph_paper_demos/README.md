# SIGGRAPH Paper Demo Candidates

This directory collects downloaded reference videos, cropped demo clips, and prompt candidates for paper-inspired
Genesis generation tests.

The current set intentionally avoids cloth, fluid, and RL/control-heavy examples. It focuses on complete demo scenes
rather than simple paper stress tests.

Coverage:

- `pure_rigid`: rigid mechanisms, chains, gears, and contact-driven assemblies.
- `soft_one_way_rigid`: deformable bodies interacting with fixed or scripted rigid boundaries.
- `rigid_soft_two_way`: rigid and deformable bodies that should mutually influence each other through contact.

Files:

- `cases.txt`: suite-style `title|prompt` entries, sorted alphabetically by title.
- `layouts/`: optional source-derived layout JSON files referenced from prompts via `@layout <relative-path>`. Layouts
  may include `reusable_assets` entries for original local/GitHub meshes and textures; those files are reused verbatim
  after one read-only mesh sanity check.
- `catalog.json`: source papers, original video paths, clipped segment paths, prompt text, and feasibility notes.
- `videos/`: downloaded full MP4 reference videos.
- `clips/`: cropped MP4 snippets named after each example title.
- `clip_contact_sheets/`: sampled frame sheets generated from the cropped snippets.
- `contact_sheets/` and `timelines/`: broader inspection sheets for the full source videos.

Notes:

- The YouTube downloads use progressive MP4 files because the workstation did not have system `ffmpeg` installed during
  download. The cropped snippets were generated with the `imageio-ffmpeg` binary through `uv run`.
- Codimensional shell/cloth/rod-heavy demos from C-IPC are left out for now because the current pipeline does not
  support cloth directly.
