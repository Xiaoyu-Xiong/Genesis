# Sorted SIGGRAPH-Style Demo Candidates

This directory reorganizes moderate-complexity paper-video examples into boundary/body-type buckets for Genesis
generation tests. The target is the same spirit as `soft_toys_squeeze_through_tube`: contact-rich enough to be useful,
but not so delicate that it depends on exact source geometry, exact placement, or heavy per-scene tuning.

Selection rules:

- Include only rigid or articulated rigid bodies, volumetric/solid deformable bodies, and fixed or scripted rigid
  boundaries.
- Exclude cloth, hair, fluids, granular media, thin rods/cables, pneumatic soft actuators, and fragile
  precision-placement scenes.
- Avoid examples like `card_house_topple`, `chainmail_net_ball_drop`, and `halloween_pumpkin_party` because they are
  respectively too placement-sensitive, too topology-sensitive, or too broad/stylized for this suite.
- The user-reviewed rejected examples have been removed: `spiky_soft_balls_pinball`, `woven_basket_soft_impact`,
  `rolling_cylinder_pin_field`, `wrecking_ball_block_pile`, `soft_lattice_cube_press`, `soft_gripper_rigid_ball`, and
  the previous moving-rigid-only mechanism set.
- Treat "boundary" as fixed or prescribed geometry. Dynamic rigid/articulated bodies listed in the category name are
  contact participants beyond that boundary.

Folders:

- `fixed_boundary_rigid_only/`
- `moving_boundary_rigid_only/`
- `fixed_boundary_deformable_only/`
- `moving_boundary_deformable_only/`
- `fixed_boundary_rigid_deformable/`
- `moving_boundary_rigid_deformable/`

Each folder contains:

- `cases.txt`: suite-style `title|prompt` entries.
- `clips/`: clipped reference videos.
- `clip_contact_sheets/`: quick visual summaries of the clips.

Top-level files:

- `catalog.json`: source videos, source segments, category labels, clip paths, contact-sheet paths, and fit notes.
- `cases_active.txt`: a roll-up of all final selected cases for convenience.
- `videos/`: downloaded external source videos that were not already present in `../siggraph_paper_demos/videos/`.

Notes:

- Rigid-only buckets are intentionally sparse after the stricter filtering, because most visually interesting pure
  rigid SIGGRAPH examples rely on chains, gears, card stacks, pin fields, or other precise layouts.
- New clips were added from Adaptive Rigidification, Adaptive Merging, Contact-Centric Deformation Learning, Trading
  Spaces, and the SCA two-way rigid/deformable coupling video, in addition to the existing local source-video set.
