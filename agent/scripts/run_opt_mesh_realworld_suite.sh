#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-gpt-5.4}"
CRITIC_MODEL="${CRITIC_MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
CRITIC_REASONING_EFFORT="${CRITIC_REASONING_EFFORT:-}"
CRITIC_PROMPT_VARIANT="${CRITIC_PROMPT_VARIANT:-full}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
BACKEND="${BACKEND:-gpu}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
XML_MAX_ATTEMPTS="${XML_MAX_ATTEMPTS:-4}"
SAMPLE_EVERY_SEC="${SAMPLE_EVERY_SEC:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-24}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1000}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_mesh_realworld_suite/${RUN_TS}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --critic-model)
      CRITIC_MODEL="$2"
      shift 2
      ;;
    --reasoning-effort)
      REASONING_EFFORT="$2"
      shift 2
      ;;
    --critic-reasoning-effort)
      CRITIC_REASONING_EFFORT="$2"
      shift 2
      ;;
    --critic-prompt-variant)
      CRITIC_PROMPT_VARIANT="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --gpu)
      BACKEND="gpu"
      shift
      ;;
    --cpu)
      BACKEND="cpu"
      shift
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--model MODEL] [--critic-model MODEL] [--reasoning-effort EFFORT] [--critic-reasoning-effort EFFORT] [--critic-prompt-variant full|compact] [--max-parallel N] [--gpu|--cpu|--backend cpu|gpu] [--run-root PATH]" >&2
      exit 2
      ;;
  esac
done

if [[ "$BACKEND" != "cpu" && "$BACKEND" != "gpu" ]]; then
  echo "Invalid BACKEND: $BACKEND" >&2
  exit 2
fi

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

run_cmd() {
  echo "+ $*"
  "$@"
}

mkdir -p "$RUN_ROOT"
TASK_FILE="$RUN_ROOT/tasks.txt"

cat > "$TASK_FILE" <<'CASES'
warehouse_loading_bay|Create a believable warehouse loading scene that feels like an active fulfillment area, with handling equipment, movable freight, and the kinds of surrounding props that naturally appear around a loading zone. Over 10s, the scene should show an ongoing loading or handling activity with plausible object interactions.

airport_service_apron|Create a realistic airport ground-service staging scene near an aircraft turnaround area, with the kinds of support equipment, movable assets, and safety infrastructure that would naturally appear there. Over 10s, the scene should depict an ongoing service operation with believable motion and contact-rich activity.

construction_logistics_corner|Create a realistic construction or site-logistics corner with active machinery, stored materials, temporary safety infrastructure, and other movable worksite elements arranged naturally. Over 10s, the environment should feel active, with a plausible equipment-led interaction sequence affecting nearby objects.

factory_packing_cell|Create a believable factory packing or packaging cell with an active automation setup operating within a realistic production environment. Over 10s, the scene should show a coherent open-loop work routine while the surrounding work area feels like a real working station.

dual_robot_workcell|Create a realistic shared automation workcell centered on a common workspace in a believable industrial environment. Over 10s, multiple active systems should move through a coordinated-looking routine while the surrounding fixtures, props, and movable items establish a rich scene.

retail_backroom_scene|Create a believable retail backroom or stockroom scene with the kinds of movable storage, handling equipment, utility props, and safety objects that would naturally accumulate in a busy support area. Over 10s, the scene should include a plausible handling or movement sequence that creates multi-object interactions.

workshop_makerspace_scene|Create a realistic workshop or makerspace scene with varied movable equipment, storage, and work-area elements arranged in a naturally cluttered but usable environment. Over 10s, the scene should show a believable disturbance or handling event that causes a rich cascade of contacts among objects.

lab_handling_station|Create a believable lab or medical handling environment with one active handling system working within a detailed tabletop or bench-top station. Over 10s, the scene should feel like an ongoing real-world handling routine, with surrounding equipment and support props contributing to a realistic environment.
CASES

echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Critic model: ${CRITIC_MODEL:-<same as model>}"
echo "Reasoning effort: ${REASONING_EFFORT:-<default>}"
echo "Critic reasoning effort: ${CRITIC_REASONING_EFFORT:-<same as generator>}"
echo "Critic prompt variant: ${CRITIC_PROMPT_VARIANT}"
echo "Max parallel tasks: ${MAX_PARALLEL}"
echo "Backend: $BACKEND"
echo "Python: ${PYTHON_CMD[*]}"

opt_cmd=(
  "${PYTHON_CMD[@]}" -m agent.opt.cli optimize-batch
  --tasks-file "$TASK_FILE"
  --model "$MODEL"
  --backend "$BACKEND"
  --max-parallel "$MAX_PARALLEL"
  --max-opt-rounds "$MAX_OPT_ROUNDS"
  --max-attempts "$MAX_ATTEMPTS"
  --xml-max-attempts "$XML_MAX_ATTEMPTS"
  --timeout-sec "$TIMEOUT_SEC"
  --sample-every-sec "$SAMPLE_EVERY_SEC"
  --max-frames "$MAX_FRAMES"
  --out-dir "$RUN_ROOT"
  --out "$RUN_ROOT/summary.json"
)
if [[ -n "$CRITIC_MODEL" ]]; then
  opt_cmd+=(--critic-model "$CRITIC_MODEL")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  opt_cmd+=(--reasoning-effort "$REASONING_EFFORT")
fi
if [[ -n "$CRITIC_REASONING_EFFORT" ]]; then
  opt_cmd+=(--critic-reasoning-effort "$CRITIC_REASONING_EFFORT")
fi
if [[ -n "$CRITIC_PROMPT_VARIANT" ]]; then
  opt_cmd+=(--critic-prompt-variant "$CRITIC_PROMPT_VARIANT")
fi

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
