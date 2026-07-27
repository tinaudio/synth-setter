#!/bin/bash
# Launch a project-pinned Codex PR-review role.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
launcher="${repo_root}/agent/_shared/run_codex_review_agent.py"
exec uv run --no-project --script "${launcher}" "$@"
