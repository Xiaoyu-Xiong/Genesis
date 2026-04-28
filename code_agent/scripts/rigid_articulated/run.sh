#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

run_code_agent_suite "rigid_articulated" "$SCRIPT_DIR/cases.txt" "$@"
