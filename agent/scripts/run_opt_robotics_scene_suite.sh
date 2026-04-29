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
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_robotics_scene_suite/${RUN_TS}}"

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
tabletop_pick_place_cell|Create a robotics tabletop pick-and-place scene with one fixed-base 6-DoF arm, a parallel jaw gripper, several small rigid blocks, a source tray, and a target tray. Set up the scene and render a clear 6s view with simple observation or stepping.

bin_picking_workcell|Create a classic bin-picking workcell with one industrial robot arm over a rigid bin filled with mixed simple parts such as boxes, cylinders, and spheres, plus a nearby drop tray. Only set up the scene and render a clear 6s view; no grasp planning or complex control is required.

peg_in_hole_fixture|Create a peg-in-hole robotics benchmark scene with one robot arm or gripper holding a rigid cylindrical peg near a fixed plate with a matching hole or socket, plus alignment markers on the table. Only set up the scene and render a clear 6s view; no insertion controller is required.

palletizing_station|Create a palletizing station with one robot arm, a stack of small rigid cartons, a pallet with several placement slots, and a short conveyor or staging table. Only set up the scene and render a clear 6s view; no complex pick-and-place policy is required.

dual_arm_handoff_bench|Create a dual-arm handoff bench with two fixed-base robot arms facing each other across a table, one shared rigid object between them, and clearly marked left and right work zones. Only set up the scene and render a clear 6s view; no coordinated handoff controller is required.

conveyor_sorting_cell|Create a conveyor sorting robotics cell with one robot arm beside a belt, several small rigid objects on the belt, and two labeled destination bins. Only set up the scene and render a clear 6s view; no real sorting policy is required.
CASES

SUMMARY_JSON="$RUN_ROOT/summary.json"

opt_cmd=(
  "${PYTHON_CMD[@]}" -m agent.opt.cli optimize-batch
  --tasks-file "$TASK_FILE"
  --out-dir "$RUN_ROOT"
  --out "$SUMMARY_JSON"
)

echo "==> optimize-batch"
run_cmd "${opt_cmd[@]}"

echo "==> summarize OpenAI usage"
run_cmd "${PYTHON_CMD[@]}" agent/scripts/summarize_openai_usage.py \
  --run-root "$RUN_ROOT" \
  --out-json "$RUN_ROOT/openai_usage_summary.json" \
  --out-tsv "$RUN_ROOT/openai_usage_summary.tsv"

while IFS= read -r round_dir; do
  [[ -z "$round_dir" ]] && continue
  if [[ ! -f "$round_dir/ir.validated.json" ]]; then
    echo "!! skipping compile for $round_dir because ir.validated.json is missing"
    continue
  fi
  case_id="$(basename "$(dirname "$round_dir")")"
  echo "==> [$case_id] compile $(basename "$round_dir")"
  run_cmd "${PYTHON_CMD[@]}" -m agent.cli compile \
    --ir "$round_dir/ir.validated.json" \
    --out "$round_dir/compiled_genesis.py"
done < <(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'round_*' | sort)

echo "Done. Results are under: $RUN_ROOT"
