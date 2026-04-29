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
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_multiarticulated_suite/${RUN_TS}}"

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
forklift_and_scara_cell|Create one forklift-like articulated vehicle and one separate fixed-base SCARA robot, and add one pallet box and one fixed loading platform. Render an 8s open-loop industrial-cell sequence where the forklift drives and raises its fork while the SCARA arm independently moves through several poses above the platform.

dual_excavator_sweep|Create two separate excavator-style articulated arms on fixed or anchored bases facing a shared work area. Add several primitive debris objects between them. Render an 8s open-loop sequence where both excavator arms perform different sweeping motions near the debris without requiring precise coordination.

three_pendulum_gallery|Create three separate simple pendulum-like articulated mechanisms mounted along a line. Render a 6s sequence where each pendulum starts from a different initial angle or receives a different initial disturbance so they swing with visibly different phases.

dual_robot_pushers|Create two separate compact mobile pushing robots. Place two movable boxes and one fixed wall in the scene. Render an 8s open-loop sequence where each robot drives toward its own box and pushes it in a different direction.

arm_pair_target_gallery|Create two separate fixed-base articulated arms, plus several fixed target spheres arranged in space. Render an 8s open-loop sequence where each arm moves through its own sequence of poses near different targets.

mobile_robot_and_excavator|Create one separate differential-drive mobile robot and one separate excavator-style articulated arm on a fixed base. Add a few primitive obstacles and one movable box. Render an 8s open-loop scene where the mobile robot drives through the environment while the excavator arm independently sweeps near the box.
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
