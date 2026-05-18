#!/usr/bin/env bash
# Compatibility shim. Shared hook helpers live in agent/hooks/_lib.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
source "${SCRIPT_DIR}/../../agent/hooks/_lib.sh"
