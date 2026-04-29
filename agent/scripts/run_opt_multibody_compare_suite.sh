#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-gpt-5.4}"
CRITIC_MODEL="${CRITIC_MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
CRITIC_REASONING_EFFORT="${CRITIC_REASONING_EFFORT:-}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
BACKEND="${BACKEND:-gpu}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
XML_MAX_ATTEMPTS="${XML_MAX_ATTEMPTS:-4}"
SAMPLE_EVERY_SEC="${SAMPLE_EVERY_SEC:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-24}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_multibody_compare_suite/${RUN_TS}}"

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
      echo "Usage: $0 [--model MODEL] [--critic-model MODEL] [--reasoning-effort EFFORT] [--critic-reasoning-effort EFFORT] [--max-parallel N] [--gpu|--cpu|--backend gpu|cpu] [--run-root PATH]" >&2
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

compile_variant_rounds() {
  local variant_root="$1"
  local round_dir=""
  while IFS= read -r round_dir; do
    [[ -z "$round_dir" ]] && continue
    local case_id
    case_id="$(basename "$(dirname "$round_dir")")"
    echo "==> [$case_id] compile $(basename "$round_dir")"
    run_cmd "${PYTHON_CMD[@]}" -m agent.cli compile \
      --ir "$round_dir/ir.validated.json" \
      --out "$round_dir/compiled_genesis.py"
  done < <(find "$variant_root" -mindepth 2 -maxdepth 2 -type d -name 'round_*' | sort)
}

run_variant() {
  local variant="$1"
  local variant_root="$RUN_ROOT/$variant"
  mkdir -p "$variant_root"

  local opt_cmd=(
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
    --critic-prompt-variant "$variant"
    --out-dir "$variant_root"
    --out "$variant_root/summary.json"
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

  echo "==> optimize-batch [$variant]"
  run_cmd "${opt_cmd[@]}"
  compile_variant_rounds "$variant_root"
}

mkdir -p "$RUN_ROOT"
TASK_FILE="$RUN_ROOT/tasks.txt"

echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Critic model: ${CRITIC_MODEL:-<same as model>}"
echo "Reasoning effort: ${REASONING_EFFORT:-<default>}"
echo "Critic reasoning effort: ${CRITIC_REASONING_EFFORT:-<same as generator>}"
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
dual_scara_shared_table|Create two separate SCARA robots mounted on opposite sides of the same table, and place one small box near the center of the table. Render an 8s open-loop sequence where the two arms move through different target poses above and around the center region without grasping.
forklift_and_scara_cell|Create one forklift-like articulated vehicle and one separate fixed-base SCARA robot, and add one pallet box and one fixed loading platform. Render an 8s open-loop industrial-cell sequence where the forklift drives and raises its fork while the SCARA arm independently moves through several poses above the platform.
dual_excavator_sweep|Create two separate excavator-style articulated arms on fixed or anchored bases facing a shared work area. Add several primitive debris objects between them. Render an 8s open-loop sequence where both excavator arms perform different sweeping motions near the debris without requiring precise coordination.
CASES

run_variant full
run_variant compact

echo "Done. Results are under: $RUN_ROOT"
