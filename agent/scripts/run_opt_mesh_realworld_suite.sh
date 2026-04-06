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
MAX_PARALLEL="${MAX_PARALLEL:-6}"
BACKEND="${BACKEND:-gpu}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
XML_MAX_ATTEMPTS="${XML_MAX_ATTEMPTS:-4}"
SAMPLE_EVERY_SEC="${SAMPLE_EVERY_SEC:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-24}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

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
warehouse_loading_bay|Create a busy warehouse loading bay with a forklift-like articulated vehicle, several palletized cargo containers, a few movable safety barriers, and a loading platform. Over 10s, the vehicle should move into the area and cause realistic contact interactions among the loose objects.

factory_packing_cell|Create a factory packing cell with a fixed-base articulated robot working around several matching storage totes, workpiece trays, and a tool cart on and around a packing table. Over 10s, the robot should move through a believable open-loop routine while the surrounding props remain part of the scene.

dual_robot_assembly_zone|Create a dual-robot assembly zone with two compact articulated robots facing a shared worktable, several matching bins and trays, and a few larger shop-floor props around the station. Over 10s, both robots should execute open-loop motions in the shared space while the environment feels like a real work cell.

workshop_clutter_rollthrough|Create a cluttered workshop scene with stools, toolboxes, storage crates, bins, and other movable shop props scattered in a believable arrangement. Over 10s, one heavy moving object should travel through the area and trigger a chain of contacts across multiple props.

construction_site_corner|Create a small construction-site corner with an articulated machine, stacked material containers, portable barriers, and other rugged site props arranged around a work zone. Over 10s, the machine should move through the scene and disturb nearby objects in a plausible way.

lab_handling_station|Create a tabletop lab handling station with one articulated arm, several trays, racks, storage bins, and a few larger support props around the workstation. Over 10s, the arm should carry out an open-loop motion sequence above the station while the props define a realistic working environment.

retail_backroom_scene|Create a retail backroom or stockroom scene with rolling containers, storage bins, stacked cases, safety obstacles, and a few larger utility props laid out like a real storage area. Over 10s, a moving object should pass through and create a complex multi-object interaction sequence.

airport_service_corner|Create an airport ground-service corner with baggage containers, safety barriers, service carts, and one articulated service vehicle or robot arranged near a staging area. Over 10s, the active articulated asset should move through the environment and produce believable interactions with the surrounding movable props.
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
