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
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_mesh_suite/${RUN_TS}}"

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
dual_mesh_barriers_break|Create two matching road safety barriers with tapered sides, recessed feet, and molded details standing on the ground with a gap between them, and launch one dense box projectile through the gap so both barriers are struck and topple over 6s.

mesh_furniture_obstacle_course|Create two matching workshop stools with round seats, angled legs, and a circular lower brace, plus one toolbox with a top handle, and one sphere that rolls through them and causes contact interactions over 6s.

forklift_with_mesh_pallet_box|Create one forklift-like articulated vehicle, two matching pallet boxes with pockets and pallet-style feet, and one fixed loading platform. Render an 8s open-loop sequence where the forklift drives toward the pallet boxes and nudges them without grasping.

scara_with_mesh_workpiece_array|Create one fixed-base SCARA robot, three identical industrial workpiece trays with raised lips, rounded corners, and shallow compartments on a table, and one small box near the trays. Render an 8s open-loop sequence where the SCARA arm moves above and around the tray region without grasping.

dual_robot_mesh_cell|Create two separate compact articulated robots on fixed bases facing a shared table, two identical storage bins with lips and handles, and one toolbox with a top handle on the table. Render an 8s open-loop scene where both articulated robots move above the table while interacting visually with the props but without grasping.

mesh_arch_and_barriers_scene|Create one decorative arch-like prop with a clear central opening and layered molding, and two identical molded safety barriers on the ground. Launch a dense sphere so it first hits one barrier and then the arch region over 6s.
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
