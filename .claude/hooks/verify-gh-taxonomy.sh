#!/usr/bin/env bash
# Compatibility shim. Shared hook implementation lives in agent/hooks/verify-gh-taxonomy.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../agent/hooks/verify-gh-taxonomy.sh"
