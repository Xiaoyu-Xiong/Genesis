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
MAX_PARALLEL="${MAX_PARALLEL:-4}"
BACKEND="${BACKEND:-gpu}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
XML_MAX_ATTEMPTS="${XML_MAX_ATTEMPTS:-4}"
SAMPLE_EVERY_SEC="${SAMPLE_EVERY_SEC:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-24}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_multibody_suite/${RUN_TS}}"

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
      echo "Usage: $0 [--model MODEL] [--critic-model MODEL] [--reasoning-effort EFFORT] [--critic-reasoning-effort EFFORT] [--critic-prompt-variant full|compact] [--max-parallel N] [--gpu|--cpu|--backend gpu|cpu] [--run-root PATH]" >&2
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

if [[ -n "$LOCAL_PYTHON" ]]; then
  PYTHON_CMD=("$LOCAL_PYTHON")
else
  PYTHON_CMD=(uv run python)
fi

run_cmd() {
  echo "+ $*"
  "$@"
}

mkdir -p "$RUN_ROOT"
TASK_FILE="$RUN_ROOT/tasks.txt"

echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Critic model: ${CRITIC_MODEL:-<same as model>}"
echo "Reasoning effort: ${REASONING_EFFORT:-<default>}"
echo "Critic reasoning effort: ${CRITIC_REASONING_EFFORT:-<same as generator>}"
echo "Critic prompt variant: ${CRITIC_PROMPT_VARIANT}"
echo "Max parallel tasks: ${MAX_PARALLEL}"
echo "Backend: $BACKEND"
echo "Python: ${PYTHON_CMD[*]}"

cat > "$TASK_FILE" <<'CASES'
box_pyramid_impact|Create a compact pyramid of small boxes resting on the ground, then launch one dense sphere at high speed into the lower layer so the pyramid collapses and scatters over 6s.

sphere_rack_break|Create a tightly packed triangular rack of equal small spheres on the ground and launch one cue sphere at high speed into the rack to produce a billiards-style break over 6s.

box_wall_breach|Create a short wall built from several stacked boxes and launch a heavy rectangular projectile into the center so the wall breaks apart and collapses over 6s.
block_arch_collapse|Create a simple freestanding arch from rectangular blocks on the ground, then send a fast small sphere into one side so the arch loses support and collapses over 6s.

mixed_bodies_on_slope|Create a fixed inclined plane with several spheres, boxes, and cylinders initially resting near the top, then let them move down the slope together, colliding, sliding, and rolling into each other over 6s.

cylinder_cluster_scatter|Create a dense cluster of upright cylinders on the ground and launch one fast box through the cluster so the cylinders topple and scatter over 6s.

offset_box_stack_settle|Create a tall stack of boxes with small alternating horizontal offsets so the stack is near the edge of stability, then simulate the settling and eventual toppling behavior over 6s.

rubble_pile_impact|Create an irregular pile of mixed boxes and cylinders on the ground and launch one dense sphere into the pile so the bodies rearrange, topple, and scatter over 6s.

bin_fill_scatter|Create a fixed open-top bin from several static box walls, place a group of small spheres above it, and let them fall into the bin while colliding with each other and with the walls over 6s.

plank_bridge_failure|Create a narrow bridge made from several thin boxes spanning between two fixed supports, then drop a dense sphere onto the bridge so the planks shift and the bridge collapses over 6s.
CASES

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
