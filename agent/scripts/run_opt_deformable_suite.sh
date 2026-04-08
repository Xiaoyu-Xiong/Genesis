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
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_deformable_suite/${RUN_TS}}"

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
soft_ball_box_fill|Create a scene where around ten small soft balls drop into an open box container over 10s.

jelly_cube_stack|Create a stylized scene where a tall stack of 10 soft cubes wobbles, leans, and collapses over 10s.

soft_cylinder_forest|Create a visually rich scene with 10 soft upright cylinders arranged like a toy obstacle field, and let one significantly heavier rigid primitive object sweep through them over 10s.

soft_ball_ramp_cascade|Create a playful ramp-and-bin scene where 10 soft balls roll, drop, and pile up inside a boxed area over 10s.

soft_blocks_and_rigid_bumpers|Create a compact arena filled with 10 soft blocks and a few significantly heavier rigid bumpers over 10s.

soft_columns_under_plate|Create a stylized balancing scene where a rigid plate or box settles onto a cluster of soft columns that visibly buckle, spread, and compress over 10s.

mixed_soft_drop_pile|Create a colorful toy-like scene where 10 soft cubes and soft balls are dropped together with a few significantly heavier rigid primitives so the whole pile churns, compresses, and reshapes over 10s.

soft_bumper_ring|Create a graphics-focused scene with a ring or semicircle of soft bumpers and launch a few dense spheres through it so the motion becomes a rich chain of wobbling contacts and deformation over 10s.
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
