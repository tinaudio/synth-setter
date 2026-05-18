#!/usr/bin/env bash
# Compatibility shim. Shared hook implementation lives in agent/hooks/doc-drift.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../agent/hooks/doc-drift.sh"
