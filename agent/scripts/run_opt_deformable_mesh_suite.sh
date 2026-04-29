#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LOCAL_PYTHON=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  LOCAL_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

if [[ -n "$LOCAL_PYTHON" ]]; then
  PYTHON_CMD=("$LOCAL_PYTHON")
else
  PYTHON_CMD=(uv run python)
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_deformable_mesh_suite/${RUN_TS}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--run-root PATH]" >&2
      exit 2
      ;;
  esac
done

run_cmd() {
  echo "+ $*"
  "$@"
}

mkdir -p "$RUN_ROOT"
TASK_FILE="$RUN_ROOT/tasks.txt"

echo "Run root: $RUN_ROOT"
echo "Python: ${PYTHON_CMD[*]}"

cat > "$TASK_FILE" <<'CASES'
stanford_bunny_drop|Create a scene where 10 identical medium-sized soft Stanford Bunnies drop from a height into an open box container with visible collision and deformation, render 10s behavior

soft_monster_stack|Create a stylized scene where a tall stack of 10 medium-sized identical soft monster toys wobbles, leans, collapses, and deforms, render 10s behavior

rubber_duck_ramp_cascade|Create a playful ramp-and-bin scene where around 10 identical medium-sized soft rubber ducks slide, tumble, deform, and pile into a boxed area, render 10s behavior

traffic_cone_obstacle_sweep|Create a scene with around 10 identical medium-sized upright soft traffic cones arranged like an obstacle field, and let one heavier rigid primitive object sweep through them, render 10s behavior

mixed_plush_drop_pile|Create a scene where several identical medium-sized soft plush toys and a few heavier rigid toy blocks are dropped together so the pile churns and reshapes, render 10s behavior

mushroom_plate_support|Create a scene where a rigid plate or box settles onto a cluster of soft mushroom-shaped support meshes, render 10s behavior

teapot_corridor_cascade|Create a narrow corridor filled with around 10 identical soft Stanford teapots so a dense rigid block pushes through them and causes a cascading tumble over 10s

animal_mesh_mix|Create a scene with around 10 identical soft animal toys with varied silhouettes dropped into a shallow arena so they collide, tumble, and pile up, render 10s behavior

toy_mesh_jumble|Create a scene with around 10 identical toy-like soft mesh props such as ducks, stars, bears, and cars packed into a pen and disturbed by one heavier rigid object over 10s
CASES

opt_cmd=(
  "${PYTHON_CMD[@]}" -m agent.opt.cli optimize-batch
  --tasks-file "$TASK_FILE"
  --out-dir "$RUN_ROOT"
  --out "$RUN_ROOT/summary.json"
)

echo "==> optimize-batch"
run_cmd "${opt_cmd[@]}"

while IFS= read -r round_dir; do
  [[ -z "$round_dir" ]] && continue
  case_id="$(basename "$(dirname "$round_dir")")"
  echo "==> [$case_id] compile $(basename "$round_dir")"
  run_cmd "${PYTHON_CMD[@]}" -m agent.cli compile \
    --ir "$round_dir/ir.validated.json" \
    --out "$round_dir/compiled_genesis.py"
done < <(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'round_*' | sort)

echo "Done. Results are under: $RUN_ROOT"
