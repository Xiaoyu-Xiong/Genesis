#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MESH_FORMAT="${MESH_FORMAT:-obj}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-2}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"
AI_MODEL="${AI_MODEL:-latest}"
ART_STYLE="${ART_STYLE:-realistic}"
SHOULD_REMESH="${SHOULD_REMESH:-1}"
TOPOLOGY="${TOPOLOGY:-triangle}"
TARGET_POLYCOUNT="${TARGET_POLYCOUNT:-8000}"
SYMMETRY_MODE="${SYMMETRY_MODE:-auto}"
MIN_COMPONENT_FACES="${MIN_COMPONENT_FACES:-100}"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-4}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
MODERATION="${MODERATION:-0}"
AUTO_SIZE="${AUTO_SIZE:-0}"
ORIGIN_AT="${ORIGIN_AT:-}"
EXTRA_PAYLOAD="${EXTRA_PAYLOAD:-}"
SKIP_POSTPROCESS="${SKIP_POSTPROCESS:-0}"
KEEP_LARGEST_COMPONENT="${KEEP_LARGEST_COMPONENT:-1}"
FTETWILD_EDGE_LENGTH_FAC="${FTETWILD_EDGE_LENGTH_FAC:-0.05}"
FTETWILD_EDGE_LENGTH_ABS="${FTETWILD_EDGE_LENGTH_ABS:-}"
FTETWILD_NO_OPTIMIZE="${FTETWILD_NO_OPTIMIZE:-0}"
FTETWILD_NO_SIMPLIFY="${FTETWILD_NO_SIMPLIFY:-0}"
FTETWILD_EPSILON="${FTETWILD_EPSILON:-0.001}"
FTETWILD_STOP_ENERGY="${FTETWILD_STOP_ENERGY:-10.0}"
FTETWILD_COARSEN="${FTETWILD_COARSEN:-0}"
FTETWILD_NUM_THREADS="${FTETWILD_NUM_THREADS:-0}"
FTETWILD_NUM_OPT_ITER="${FTETWILD_NUM_OPT_ITER:-80}"
FTETWILD_DISABLE_FILTERING="${FTETWILD_DISABLE_FILTERING:-0}"
FTETWILD_VERBOSE="${FTETWILD_VERBOSE:-0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/mesh_meshy_suite/${RUN_TS}}"
TASKS_FILE="${TASKS_FILE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mesh-format)
      MESH_FORMAT="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --poll-interval-sec)
      POLL_INTERVAL_SEC="$2"
      shift 2
      ;;
    --max-wait-sec)
      MAX_WAIT_SEC="$2"
      shift 2
      ;;
    --ai-model)
      AI_MODEL="$2"
      shift 2
      ;;
    --art-style)
      ART_STYLE="$2"
      shift 2
      ;;
    --should-remesh)
      SHOULD_REMESH="1"
      shift
      ;;
    --topology)
      TOPOLOGY="$2"
      shift 2
      ;;
    --target-polycount)
      TARGET_POLYCOUNT="$2"
      shift 2
      ;;
    --symmetry-mode)
      SYMMETRY_MODE="$2"
      shift 2
      ;;
    --min-component-faces)
      MIN_COMPONENT_FACES="$2"
      shift 2
      ;;
    --max-repair-attempts)
      MAX_REPAIR_ATTEMPTS="$2"
      shift 2
      ;;
    --negative-prompt)
      NEGATIVE_PROMPT="$2"
      shift 2
      ;;
    --moderation)
      MODERATION="1"
      shift
      ;;
    --auto-size)
      AUTO_SIZE="1"
      shift
      ;;
    --origin-at)
      ORIGIN_AT="$2"
      shift 2
      ;;
    --extra-payload)
      EXTRA_PAYLOAD="$2"
      shift 2
      ;;
    --skip-postprocess)
      SKIP_POSTPROCESS="1"
      shift
      ;;
    --keep-largest-component)
      KEEP_LARGEST_COMPONENT="1"
      shift
      ;;
    --ftetwild-edge-length-fac)
      FTETWILD_EDGE_LENGTH_FAC="$2"
      shift 2
      ;;
    --ftetwild-edge-length-abs)
      FTETWILD_EDGE_LENGTH_ABS="$2"
      shift 2
      ;;
    --ftetwild-no-optimize)
      FTETWILD_NO_OPTIMIZE="1"
      shift
      ;;
    --ftetwild-no-simplify)
      FTETWILD_NO_SIMPLIFY="1"
      shift
      ;;
    --ftetwild-epsilon)
      FTETWILD_EPSILON="$2"
      shift 2
      ;;
    --ftetwild-stop-energy)
      FTETWILD_STOP_ENERGY="$2"
      shift 2
      ;;
    --ftetwild-coarsen)
      FTETWILD_COARSEN="1"
      shift
      ;;
    --ftetwild-num-threads)
      FTETWILD_NUM_THREADS="$2"
      shift 2
      ;;
    --ftetwild-num-opt-iter)
      FTETWILD_NUM_OPT_ITER="$2"
      shift 2
      ;;
    --ftetwild-disable-filtering)
      FTETWILD_DISABLE_FILTERING="1"
      shift
      ;;
    --ftetwild-verbose)
      FTETWILD_VERBOSE="1"
      shift
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --tasks-file)
      TASKS_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--mesh-format obj|glb|stl] [--timeout-sec SEC] [--poll-interval-sec SEC] [--max-wait-sec SEC] [--ai-model latest|meshy-6|meshy-5] [--art-style realistic|sculpture] [--should-remesh] [--topology triangle|quad] [--target-polycount N] [--symmetry-mode off|auto|on] [--min-component-faces N] [--max-repair-attempts N] [--ftetwild-edge-length-fac X] [--ftetwild-edge-length-abs X] [--ftetwild-no-optimize] [--ftetwild-no-simplify] [--ftetwild-epsilon X] [--ftetwild-stop-energy X] [--ftetwild-coarsen] [--ftetwild-num-threads N] [--ftetwild-num-opt-iter N] [--ftetwild-disable-filtering] [--ftetwild-verbose] [--negative-prompt TEXT] [--moderation] [--auto-size] [--origin-at bottom|center] [--extra-payload JSON_OR_PATH] [--skip-postprocess] [--keep-largest-component] [--run-root PATH] [--tasks-file PATH]" >&2
      exit 2
      ;;
  esac
done

if [[ "$MESH_FORMAT" != "obj" && "$MESH_FORMAT" != "glb" && "$MESH_FORMAT" != "stl" ]]; then
  echo "Invalid MESH_FORMAT: $MESH_FORMAT" >&2
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

if [[ -z "$TASKS_FILE" ]]; then
  TASKS_FILE="$RUN_ROOT/tasks.txt"
  cat > "$TASKS_FILE" <<'CASES'
industrial_storage_crate|Create a single sturdy industrial storage crate with recessed handles, a flat base, and a practical hard-surface silhouette suitable for rigid-body simulation.
metal_toolbox|Create a single compact metal toolbox with a rectangular body, a hinged lid silhouette, and a stable flat bottom suitable for rigid-body simulation.
warehouse_pallet_jack|Create a single simplified pallet-jack-like hard-surface object with clean geometry, a stable footprint, and no tiny fragile details.
safety_barrier|Create a single freestanding road safety barrier with a broad stable base and clean hard-surface geometry suitable for rigid-body simulation.
shipping_case|Create a single rugged shipping case with reinforced corners, a flat resting base, and simple hard-surface geometry suitable for rigid-body simulation.
workshop_stool|Create a single industrial workshop stool with a broad seat, sturdy legs, and geometry simplified for rigid-body simulation.
traffic_cone|Create a single traffic cone with a heavy square base and a simple clean silhouette suitable for rigid-body simulation.
concrete_block|Create a single concrete construction block with recessed holes and slightly beveled hard edges, keeping the geometry clean and simulation-friendly.
foam_cushion_block|Create a single soft foam cushion block with rounded edges and a simple closed silhouette suitable for deformable-body simulation.
silicone_squeeze_bottle|Create a single soft silicone squeeze bottle with a broad stable base, a simple narrow neck, and a clean closed outer shell suitable for deformable-body simulation.
gel_ice_pack|Create a single sealed gel ice pack pouch with a rounded rectangular shape and smooth closed boundaries, suitable for deformable-body simulation.
stress_ball|Create a single soft stress ball with a simple rounded shape and no thin appendages, keeping the geometry clean and closed for deformable-body simulation.
rubber_eraser_block|Create a single soft rubber eraser block with rounded edges, a simple solid silhouette, and geometry suitable for deformable-body simulation.
foam_packing_insert|Create a single foam packing insert block with a few broad recessed cavities and a clean closed outer shell suitable for deformable-body simulation.
silicone_hot_water_bottle|Create a single rubber hot-water-bottle-like object with a rounded rectangular body and a simple cap silhouette, keeping the geometry clean and closed for deformable-body simulation.
CASES
fi

SUMMARY_TSV="$RUN_ROOT/summary.tsv"
FAILED_TXT="$RUN_ROOT/failed_cases.txt"
: > "$FAILED_TXT"
printf "case_id\tstatus\traw_manifold_ok\trepaired_manifold_ok\traw_vertices\traw_faces\tfinal_vertices\tfinal_faces\twait_preview_sec\trepair_total_sec\tcase_dir\tresult_json\n" > "$SUMMARY_TSV"

echo "Run root: $RUN_ROOT"
echo "Tasks file: $TASKS_FILE"
echo "Mesh format: $MESH_FORMAT"
echo "AI model: $AI_MODEL"
echo "Art style: $ART_STYLE"
echo "Should remesh: $([[ "$SHOULD_REMESH" == "1" ]] && echo "yes" || echo "no")"
echo "Topology: $TOPOLOGY"
echo "Target polycount: ${TARGET_POLYCOUNT:-<unset>}"
echo "Symmetry mode: $SYMMETRY_MODE"
echo "Min component faces: $MIN_COMPONENT_FACES"
echo "Max repair attempts: $MAX_REPAIR_ATTEMPTS"
echo "Timeout: ${TIMEOUT_SEC}s"
echo "Poll interval: ${POLL_INTERVAL_SEC}s"
echo "Max wait: ${MAX_WAIT_SEC}s"
echo "Postprocess: $([[ "$SKIP_POSTPROCESS" == "1" ]] && echo "disabled" || echo "enabled")"
echo "Keep largest component: $([[ "$KEEP_LARGEST_COMPONENT" == "1" ]] && echo "yes" || echo "no")"
echo "fTetWild edge length fac: $FTETWILD_EDGE_LENGTH_FAC"
echo "fTetWild edge length abs: ${FTETWILD_EDGE_LENGTH_ABS:-<unset>}"
echo "fTetWild optimize: $([[ "$FTETWILD_NO_OPTIMIZE" == "1" ]] && echo "no" || echo "yes")"
echo "fTetWild simplify: $([[ "$FTETWILD_NO_SIMPLIFY" == "1" ]] && echo "no" || echo "yes")"
echo "fTetWild epsilon: $FTETWILD_EPSILON"
echo "fTetWild stop energy: $FTETWILD_STOP_ENERGY"
echo "Python: ${PYTHON_CMD[*]}"

total=0
passed=0
failed=0
raw_manifold_passed=0
repaired_manifold_passed=0

while IFS='|' read -r case_id prompt; do
  case_id="${case_id//$'\r'/}"
  prompt="${prompt//$'\r'/}"
  [[ -z "${case_id// }" ]] && continue
  [[ "$case_id" =~ ^# ]] && continue
  if [[ -z "${prompt// }" ]]; then
    echo "Skipping malformed task line for case: $case_id" >&2
    continue
  fi

  total=$((total + 1))
  case_dir="$RUN_ROOT/$case_id"
  result_json="$case_dir/result.json"
  mkdir -p "$case_dir"

  echo "==> [$case_id] generate"
  cmd=(
    "${PYTHON_CMD[@]}" -m agent.mesh generate
    --prompt "$prompt"
    --mesh-format "$MESH_FORMAT"
    --timeout-sec "$TIMEOUT_SEC"
    --poll-interval-sec "$POLL_INTERVAL_SEC"
    --max-wait-sec "$MAX_WAIT_SEC"
    --ai-model "$AI_MODEL"
    --art-style "$ART_STYLE"
    --topology "$TOPOLOGY"
    --symmetry-mode "$SYMMETRY_MODE"
    --min-component-faces "$MIN_COMPONENT_FACES"
    --max-repair-attempts "$MAX_REPAIR_ATTEMPTS"
    --ftetwild-edge-length-fac "$FTETWILD_EDGE_LENGTH_FAC"
    --ftetwild-epsilon "$FTETWILD_EPSILON"
    --ftetwild-stop-energy "$FTETWILD_STOP_ENERGY"
    --ftetwild-num-threads "$FTETWILD_NUM_THREADS"
    --ftetwild-num-opt-iter "$FTETWILD_NUM_OPT_ITER"
    --out-dir "$case_dir"
    --out "$result_json"
  )
  if [[ "$SHOULD_REMESH" == "1" ]]; then
    cmd+=(--should-remesh)
  fi
  if [[ -n "$TARGET_POLYCOUNT" ]]; then
    cmd+=(--target-polycount "$TARGET_POLYCOUNT")
  fi
  if [[ -n "$NEGATIVE_PROMPT" ]]; then
    cmd+=(--negative-prompt "$NEGATIVE_PROMPT")
  fi
  if [[ "$MODERATION" == "1" ]]; then
    cmd+=(--moderation)
  fi
  if [[ "$AUTO_SIZE" == "1" ]]; then
    cmd+=(--auto-size)
  fi
  if [[ -n "$ORIGIN_AT" ]]; then
    cmd+=(--origin-at "$ORIGIN_AT")
  fi
  if [[ -n "$EXTRA_PAYLOAD" ]]; then
    cmd+=(--extra-payload "$EXTRA_PAYLOAD")
  fi
  if [[ "$SKIP_POSTPROCESS" == "1" ]]; then
    cmd+=(--skip-postprocess)
  fi
  if [[ "$KEEP_LARGEST_COMPONENT" == "1" ]]; then
    cmd+=(--keep-largest-component)
  fi
  if [[ -n "$FTETWILD_EDGE_LENGTH_ABS" ]]; then
    cmd+=(--ftetwild-edge-length-abs "$FTETWILD_EDGE_LENGTH_ABS")
  fi
  if [[ "$FTETWILD_NO_OPTIMIZE" == "1" ]]; then
    cmd+=(--ftetwild-no-optimize)
  fi
  if [[ "$FTETWILD_NO_SIMPLIFY" == "1" ]]; then
    cmd+=(--ftetwild-no-simplify)
  fi
  if [[ "$FTETWILD_COARSEN" == "1" ]]; then
    cmd+=(--ftetwild-coarsen)
  fi
  if [[ "$FTETWILD_DISABLE_FILTERING" == "1" ]]; then
    cmd+=(--ftetwild-disable-filtering)
  fi
  if [[ "$FTETWILD_VERBOSE" == "1" ]]; then
    cmd+=(--ftetwild-verbose)
  fi

  if run_cmd "${cmd[@]}"; then
    passed=$((passed + 1))
    raw_ok="false"
    repaired_ok="false"
    raw_vertices="na"
    raw_faces="na"
    final_vertices="na"
    final_faces="na"
    wait_preview_sec="na"
    repair_total_sec="na"
    if [[ -f "$case_dir/raw_manifold_check.json" ]] && grep -q '"ok": true' "$case_dir/raw_manifold_check.json"; then
      raw_ok="true"
      raw_manifold_passed=$((raw_manifold_passed + 1))
    fi
    if [[ -f "$case_dir/manifold_check.json" ]] && grep -q '"ok": true' "$case_dir/manifold_check.json"; then
      repaired_ok="true"
      repaired_manifold_passed=$((repaired_manifold_passed + 1))
    fi
    if [[ -f "$case_dir/raw_manifold_check.json" ]]; then
      raw_vertices="$(grep -m1 '"vertex_count"' "$case_dir/raw_manifold_check.json" | sed -E 's/.*: ([0-9]+).*/\1/' || true)"
      raw_faces="$(grep -m1 '"face_count"' "$case_dir/raw_manifold_check.json" | sed -E 's/.*: ([0-9]+).*/\1/' || true)"
      [[ -z "$raw_vertices" ]] && raw_vertices="na"
      [[ -z "$raw_faces" ]] && raw_faces="na"
    fi
    if [[ -f "$case_dir/manifold_check.json" ]]; then
      final_vertices="$(grep -m1 '"vertex_count"' "$case_dir/manifold_check.json" | sed -E 's/.*: ([0-9]+).*/\1/' || true)"
      final_faces="$(grep -m1 '"face_count"' "$case_dir/manifold_check.json" | sed -E 's/.*: ([0-9]+).*/\1/' || true)"
      [[ -z "$final_vertices" ]] && final_vertices="na"
      [[ -z "$final_faces" ]] && final_faces="na"
    fi
    if [[ -f "$case_dir/profile.json" ]]; then
      wait_preview_sec="$(grep -m1 '"wait_preview"' "$case_dir/profile.json" | sed -E 's/.*: ([0-9.]+).*/\1/' || true)"
      repair_total_sec="$(grep -m1 '"repair_total"' "$case_dir/profile.json" | sed -E 's/.*: ([0-9.]+).*/\1/' || true)"
      [[ -z "$wait_preview_sec" ]] && wait_preview_sec="na"
      [[ -z "$repair_total_sec" ]] && repair_total_sec="na"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$case_id" "ok" "$raw_ok" "$repaired_ok" "$raw_vertices" "$raw_faces" "$final_vertices" "$final_faces" "$wait_preview_sec" "$repair_total_sec" "$case_dir" "$result_json" >> "$SUMMARY_TSV"
  else
    failed=$((failed + 1))
    printf "%s\n" "$case_id" >> "$FAILED_TXT"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$case_id" "failed" "false" "false" "na" "na" "na" "na" "na" "na" "$case_dir" "$result_json" >> "$SUMMARY_TSV"
    echo "!! [$case_id] failed; continuing"
  fi
done < "$TASKS_FILE"

echo
echo "Done."
echo "Total cases: $total"
echo "Passed: $passed"
echo "Failed: $failed"
echo "Raw manifold pass: $raw_manifold_passed"
echo "Repaired manifold pass: $repaired_manifold_passed"
echo "Summary: $SUMMARY_TSV"
if [[ -s "$FAILED_TXT" ]]; then
  echo "Failed cases: $FAILED_TXT"
else
  rm -f "$FAILED_TXT"
fi
echo "Results are under: $RUN_ROOT"
