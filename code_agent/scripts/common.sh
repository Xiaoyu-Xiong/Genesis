#!/usr/bin/env bash
set -euo pipefail

CODE_AGENT_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_AGENT_ROOT_DIR="$(cd "$CODE_AGENT_SCRIPTS_DIR/../.." && pwd)"

run_cmd() {
  echo "+ $*"
  "$@"
}

run_code_agent_suite() {
  local suite_name="$1"
  local cases_file="$2"
  shift 2

  local run_ts
  run_ts="$(date +%Y%m%d_%H%M%S)"
  local run_root="${RUN_ROOT:-code_agent/workspaces/suites/${suite_name}/${run_ts}}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-root)
        run_root="$2"
        shift 2
        ;;
      *)
        break
        ;;
    esac
  done

  cd "$CODE_AGENT_ROOT_DIR"
  mkdir -p "$run_root"
  cp "$cases_file" "$run_root/cases.txt"

  echo "Suite: $suite_name"
  echo "Cases: $cases_file"
  echo "Run root: $run_root"

  if [[ -n "${CODE_AGENT_CMD:-}" ]]; then
    read -r -a code_agent_cmd <<< "$CODE_AGENT_CMD"
  else
    if [[ ! -f "$CODE_AGENT_ROOT_DIR/code_agent/cli.py" ]]; then
      echo "code_agent CLI is not implemented yet." >&2
      echo "Set CODE_AGENT_CMD to an experimental command, or implement code_agent.cli run-suite." >&2
      exit 3
    fi
    code_agent_cmd=(uv run python -m code_agent.cli run-suite)
  fi

  run_cmd "${code_agent_cmd[@]}" \
    --tasks-file "$run_root/cases.txt" \
    --out-dir "$run_root" \
    "$@"
}
