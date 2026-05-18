#!/usr/bin/env bash
# Compatibility shim. Shared hook tests live in agent/hooks/test.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../agent/hooks/test.sh"
