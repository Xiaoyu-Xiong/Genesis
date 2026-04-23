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
traffic_barrel_compactor_lane|Create a scene where multiple identical soft orange traffic barrels with reflective lane markings are crowded into a box and compressed by a moving rigid compactor over 10s

watermelon_panel_squeeze|Create a scene where several identical soft striped watermelons are trapped on a low platform and squeezed sideways by two rigid panels, showing obvious bulging, flattening, and pileup over 10s

toy_bunny_gate_jam|Create a scene where several identical soft toy bunnies with bright painted details are compressed by a rigid plate over 10s

spray_bottle_crate_press|Create a scene where a group of identical soft translucent spray bottles with label texture is partially boxed in and compressed from above by a rigid plate, showing visible buckling and recovery over 10s

striped_candy_hopper_crush|Create a playful scene where many identical soft striped candy pieces slide into a shallow hopper and are then compacted by a rigid pusher, producing clear deformation, crowding, and surface texture motion over 10s

wooden_barrel_sweep_stack|Create a scene where several identical soft wooden barrels with visible wood grain and metal band texture are swept into a corner by a heavy rigid block, causing rolling, squashing, and pile reshaping over 10s

banana_bunch_press_array|Create a scene where several identical soft bananas with readable yellow peel texture are arranged in rows and compressed by a descending rigid frame, creating obvious bending, flattening, and lateral spreading over 10s
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
