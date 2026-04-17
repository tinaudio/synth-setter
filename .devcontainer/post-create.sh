#!/usr/bin/env bash
# Dev container first-run setup for both Codespaces and local devcontainers.
# Runs once after the container is created. The base image
# (tinaudio/synth-setter:dev-snapshot) already ships all deps, Surge XT,
# xvfb, and rclone — but NOT credentials. The devcontainer configs do not
# forward `.env` automatically; R2 and W&B creds must be provided at
# runtime via Codespaces secrets or other devcontainer environment-variable
# configuration, or sourced manually inside the container shell.
set -euo pipefail

# Locate the workspace root via the .project-root anchor, not by hardcoded
# path. GitHub Codespaces mounts at /workspaces/synth-setter, but locally the
# devcontainer CLI uses the host directory basename (e.g. a git worktree
# name), so the mount path is not fixed.
search_start="$(cd "$(dirname "$0")" && pwd)"
dir="$search_start"
while [[ "$dir" != "/" && ! -f "$dir/.project-root" ]]; do
  dir="$(dirname "$dir")"
done
[[ -f "$dir/.project-root" ]] || {
  echo "ERROR: .project-root anchor not found walking up from $search_start." >&2
  echo "The dev container must be opened at the repository root containing .project-root." >&2
  exit 1
}
cd "$dir"

# Codespaces runs this script as root against a workspace that may be owned
# by another UID, tripping git's safe.directory check. Mark the repo trusted
# before later git config calls and pre-commit install.
git config --global --add safe.directory "$(pwd)"

if [ -n "${RESTRICTED_AGENT_GIT_PAT:-}" ]; then
  # Strip surrounding double or single quotes if present
  # (Docker's --env-file doesn't strip them like shell `source` does)
  RESTRICTED_AGENT_GIT_PAT="${RESTRICTED_AGENT_GIT_PAT%\"}"
  RESTRICTED_AGENT_GIT_PAT="${RESTRICTED_AGENT_GIT_PAT#\"}"
  RESTRICTED_AGENT_GIT_PAT="${RESTRICTED_AGENT_GIT_PAT%\'}"
  RESTRICTED_AGENT_GIT_PAT="${RESTRICTED_AGENT_GIT_PAT#\'}"
  if printf '%s' "$RESTRICTED_AGENT_GIT_PAT" | gh auth login --with-token; then
    gh auth setup-git
    echo "Git configured with RESTRICTED_AGENT_GIT_PAT"
  else
    echo "WARNING: gh auth login failed — token may be invalid. Continuing without git credential config." >&2
  fi
else
  echo "RESTRICTED_AGENT_GIT_PAT not set, skipping git credential config"
fi

# Pre-commit hooks (pre-commit itself is in the image's deps). Strip any
# absolute host-path core.hooksPath that may leak from the host .git/config
# (harmless in Codespaces; bites local devcontainer users).
git config --local --unset-all core.hooksPath 2>/dev/null || true
pre-commit install

echo "Dev container ready. Run 'make test' to verify."
