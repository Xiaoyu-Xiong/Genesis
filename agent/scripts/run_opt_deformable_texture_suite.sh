#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LOCAL_PYTHON=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  LOCAL_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

if [[ -n "${APPTAINER_SIF:-}" ]]; then
  if [[ -n "$LOCAL_PYTHON" ]]; then
    PYTHON_CMD=(apptainer exec --nv "$APPTAINER_SIF" "$LOCAL_PYTHON")
  else
    PYTHON_CMD=(apptainer exec --nv "$APPTAINER_SIF" uv run python)
  fi
else
  if [[ -n "$LOCAL_PYTHON" ]]; then
    PYTHON_CMD=("$LOCAL_PYTHON")
  else
    PYTHON_CMD=(uv run python)
  fi
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_deformable_texture_suite/${RUN_TS}}"

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
textured_duck_plate_press|Create a scene where around 10 identical medium-sized soft rubber ducks with bright toy-like texture are compressed by a descending rigid plate, showing visible squashing, render 10s behavior

striped_gel_block_squeeze|Create a scene where several identical medium-sized soft striped gel blocks are squeezed between moving rigid paddles, render 10s behavior

gift_box_crush_stack|Create a scene where a stack of identical soft wrapped gift boxes with ribbon texture is compressed and partially toppled by a heavy rigid box over 10s

plush_drag_train|Create a scene where several identical soft plush animal toys with clear fabric-like texture are dragged across the ground by a moving rigid bar, causing sliding, stretching, and pileup over 10s

beach_ball_gate_sweep|Create a scene where identical soft striped beach balls are pushed through a narrowing rigid gate with clear deformation, render 10s behavior

cap_pull_and_release|Create a scene where a soft textured baseball cap is pulled outward by moving rigid blockers and then released, making the brim and crown visibly deform over 10s

teapot_press_corridor|Create a corridor scene packed with identical soft textured Stanford teapots while a rigid pusher drives through them, causing dense contact and visible deformation over 10s

monster_ring_compression|Create a ring of identical soft monster toys with colorful textured surfaces and let a rigid plate descend into the ring so they compress and buckle over 10s

duck_ramp_cascade_texture|Create a playful ramp-and-bin scene where identical soft textured rubber ducks slide, tumble, deform, and pile into a boxed area, render 10s behavior
CASES

SUMMARY_JSON="$RUN_ROOT/summary.json"

opt_cmd=(
  "${PYTHON_CMD[@]}" -m agent.opt.cli optimize-batch
  --tasks-file "$TASK_FILE"
  --out-dir "$RUN_ROOT"
  --out "$SUMMARY_JSON"
  --mesh-texture-enabled
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
