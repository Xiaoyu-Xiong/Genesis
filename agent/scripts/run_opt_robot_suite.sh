#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-gpt-5.4}"
CRITIC_MODEL="${CRITIC_MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
CRITIC_REASONING_EFFORT="${CRITIC_REASONING_EFFORT:-}"
BACKEND="${BACKEND:-gpu}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
XML_MAX_ATTEMPTS="${XML_MAX_ATTEMPTS:-4}"
SAMPLE_EVERY_SEC="${SAMPLE_EVERY_SEC:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-24}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_robot_suite/${RUN_TS}}"

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
      echo "Usage: $0 [--model MODEL] [--critic-model MODEL] [--reasoning-effort EFFORT] [--critic-reasoning-effort EFFORT] [--gpu|--cpu|--backend gpu|cpu] [--run-root PATH]" >&2
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

run_case() {
  local case_id="$1"
  local task="$2"

  local case_dir="$RUN_ROOT/$case_id"
  mkdir -p "$case_dir"
  printf '%s\n' "$task" > "$case_dir/task.txt"

  local opt_cmd=(
    "${PYTHON_CMD[@]}" -m agent.opt.cli optimize
    --task "$task"
    --model "$MODEL"
    --backend "$BACKEND"
    --max-opt-rounds "$MAX_OPT_ROUNDS"
    --max-attempts "$MAX_ATTEMPTS"
    --xml-max-attempts "$XML_MAX_ATTEMPTS"
    --timeout-sec "$TIMEOUT_SEC"
    --sample-every-sec "$SAMPLE_EVERY_SEC"
    --max-frames "$MAX_FRAMES"
    --out-dir "$case_dir"
    --out "$case_dir/summary.json"
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

  echo "==> [$case_id] optimize"
  run_cmd "${opt_cmd[@]}"
}

mkdir -p "$RUN_ROOT"

echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Critic model: ${CRITIC_MODEL:-<same as model>}"
echo "Reasoning effort: ${REASONING_EFFORT:-<default>}"
echo "Critic reasoning effort: ${CRITIC_REASONING_EFFORT:-<same as generator>}"
echo "Backend: $BACKEND"
echo "Python: ${PYTHON_CMD[*]}"

while IFS='|' read -r case_id task; do
  [[ -z "$case_id" ]] && continue
  [[ "${case_id:0:1}" == "#" ]] && continue
  run_case "$case_id" "$task"
done <<'CASES'
mobile_base_patrol|Create a simple wheeled mobile robot that drives forward, pauses, turns, and drives to a second waypoint with a follow camera over 8s.
forklift_maneuver|Generate a forklift-like articulated vehicle that drives forward and raises its fork assembly during motion over 8s.
delta_robot_demo|Generate a delta-robot-style articulated mechanism and move its end platform through several distinct heights over 8s.
excavator_scoop|Create an excavator-style arm on a tracked base and render a scoop-like arm motion sequence over 8s.
scara_sequence|Generate a SCARA-style robot and move through multiple target positions with clear joint-space transitions over 8s.
self_balancing_bot|Create a two-wheel self-balancing robot and render stabilization followed by a small forward motion over 8s.
CASES

echo "Done. Results are under: $RUN_ROOT"
