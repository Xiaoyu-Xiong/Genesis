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
RUN_ROOT="${RUN_ROOT:-agent/runs/mesh_meshy_texture_suite/${RUN_TS}}"

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
rubber_duck|Create a single stylized rubber duck with a bright toy-like surface appearance, clean proportions, and a clearly readable beak/body texture opportunity.

traffic_cone|Create a single traffic cone with a safety-orange body, pale reflective band region, and a dark base, keeping the shape clean and recognizable.

wooden_shipping_crate|Create a single rugged wooden shipping crate with visible wood plank character, panel variation, and reinforced hard-surface structure.

soft_monster_toy|Create a single cute soft monster toy with a simple plush-like silhouette and a visually distinctive colorful toy surface.

stanford_teapot|Create a single Stanford teapot with a clear ceramic-like surface and enough shape readability for visible texture inspection.

painted_ceramic_mug|Create a single ceramic mug with a clearly readable painted surface treatment, a visible handle, and a clean silhouette suitable for texture inspection.

striped_beach_ball|Create a single inflatable beach ball with bold colored stripe regions and a simple round silhouette that makes texture seams easy to inspect.

soccer_ball|Create a single soccer ball with clearly readable black-and-white panel patterning and a clean spherical silhouette.

toy_fire_truck|Create a single toy fire truck with a bold red painted body, distinct window regions, and simple hard-surface proportions suitable for texture inspection.

wooden_barrel|Create a single wooden barrel with visible wood slats and metal band regions, keeping the silhouette simple and clearly readable.

gift_box|Create a single wrapped gift box with a ribbon-like color contrast and crisp surface patterning that is easy to validate in renders.

striped_candy|Create a single wrapped candy with strong stripe color variation and a compact silhouette that makes texture placement easy to inspect.

basketball|Create a single basketball with clearly visible orange pebbled surface character and black seam lines.

watermelon|Create a single watermelon with a green striped rind pattern and a simple rounded shape that makes texture continuity easy to inspect.

banana|Create a single banana with a smooth yellow peel surface and subtle color variation, keeping the silhouette clean and recognizable.

traffic_barrel|Create a single road traffic barrel with bright orange body color and contrasting stripe regions, emphasizing texture readability.

spray_bottle|Create a single household spray bottle with a distinct colored label region and contrasting plastic body parts suitable for texture inspection.

baseball_cap|Create a single baseball cap with clearly separable brim and crown color regions and a clean textile-like appearance.

toy_train|Create a single toy train engine with painted panel regions, windows, and simple color blocking that is easy to verify in rendered views.
CASES

SUMMARY_TSV="$RUN_ROOT/summary.tsv"
FAILED_TXT="$RUN_ROOT/failed_cases.txt"
: > "$FAILED_TXT"
printf "case_id\tgenerate_status\ttexture_ok\trender_ok\traw_manifold_ok\trepaired_manifold_ok\tpng_count\twait_preview_sec\twait_texture_refine_sec\trepair_total_sec\tcase_dir\tresult_json\n" > "$SUMMARY_TSV"

total=0
passed=0
failed=0
render_passed=0

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
  render_json="$case_dir/render_views/result.json"
  textured_obj="$case_dir/textured/model.obj"
  mkdir -p "$case_dir"

  echo "==> [$case_id] generate textured mesh"
  if run_cmd "${PYTHON_CMD[@]}" -m agent.mesh.cli generate \
    --prompt "$prompt" \
    --generate-texture \
    --out-dir "$case_dir" \
    --out "$result_json"; then
    generate_status="ok"
  else
    generate_status="failed"
  fi

  texture_ok="false"
  render_ok="false"
  raw_ok="false"
  repaired_ok="false"
  png_count="0"
  wait_preview_sec="na"
  wait_texture_refine_sec="na"
  repair_total_sec="na"

  if [[ "$generate_status" == "ok" ]]; then
    if [[ -f "$case_dir/raw_manifold_check.json" ]] && grep -q '"ok": true' "$case_dir/raw_manifold_check.json"; then
      raw_ok="true"
    fi
    if [[ -f "$case_dir/manifold_check.json" ]] && grep -q '"ok": true' "$case_dir/manifold_check.json"; then
      repaired_ok="true"
    fi
    if [[ -f "$textured_obj" && -f "$case_dir/textured/model.mtl" && -f "$case_dir/textured/base_color.png" ]]; then
      texture_ok="true"
    fi
    if [[ -f "$case_dir/profile.json" ]]; then
      wait_preview_sec="$(grep -m1 '"wait_preview"' "$case_dir/profile.json" | sed -E 's/.*: ([0-9.]+).*/\1/' || true)"
      wait_texture_refine_sec="$(grep -m1 '"wait_texture_refine"' "$case_dir/profile.json" | sed -E 's/.*: ([0-9.]+).*/\1/' || true)"
      repair_total_sec="$(grep -m1 '"repair_total"' "$case_dir/profile.json" | sed -E 's/.*: ([0-9.]+).*/\1/' || true)"
      [[ -z "$wait_preview_sec" ]] && wait_preview_sec="na"
      [[ -z "$wait_texture_refine_sec" ]] && wait_texture_refine_sec="na"
      [[ -z "$repair_total_sec" ]] && repair_total_sec="na"
    fi

    if [[ "$texture_ok" == "true" ]]; then
      echo "==> [$case_id] render textured views"
      if run_cmd "${PYTHON_CMD[@]}" -m agent.mesh.cli render-textured-views \
        --mesh "$textured_obj" \
        --out-dir "$case_dir/render_views" \
        --out "$render_json"; then
        render_ok="true"
        render_passed=$((render_passed + 1))
      fi
    fi

    if [[ -d "$case_dir/render_views" ]]; then
      png_count="$(find "$case_dir/render_views" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
    fi

    case_pass="true"
    if [[ "$texture_ok" != "true" || "$render_ok" != "true" ]]; then
      case_pass="false"
    fi
    if [[ "$case_pass" == "true" ]]; then
      passed=$((passed + 1))
    else
      failed=$((failed + 1))
      printf "%s\n" "$case_id" >> "$FAILED_TXT"
      echo "!! [$case_id] failed; continuing"
    fi
  else
    failed=$((failed + 1))
    printf "%s\n" "$case_id" >> "$FAILED_TXT"
    echo "!! [$case_id] failed; continuing"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$case_id" "$generate_status" "$texture_ok" "$render_ok" "$raw_ok" "$repaired_ok" "$png_count" "$wait_preview_sec" "$wait_texture_refine_sec" "$repair_total_sec" "$case_dir" "$result_json" >> "$SUMMARY_TSV"
done < "$TASK_FILE"

echo
echo "Done."
echo "Total cases: $total"
echo "Passed: $passed"
echo "Failed: $failed"
echo "Render passed: $render_passed"
echo "Summary: $SUMMARY_TSV"
if [[ -s "$FAILED_TXT" ]]; then
  echo "Failed cases: $FAILED_TXT"
else
  rm -f "$FAILED_TXT"
fi
echo "Results are under: $RUN_ROOT"
