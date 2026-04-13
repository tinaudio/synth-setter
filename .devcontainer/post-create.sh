#!/usr/bin/env bash
# Dev container first-run setup for both Codespaces and local devcontainers.
# Runs once after the container is created. The base image
# (tinaudio/perm:dev-snapshot) already ships all deps, Surge XT, xvfb,
# rclone, and baked R2/W&B credentials — so this script only wires up the
# workspace checkout to the image's venv.
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
# before any git operation (submodule update, pre-commit install).
git config --global --add safe.directory "$(pwd)"

# Skills live in a git submodule (.claude/skills → tinaudio/skills).
git submodule update --init --recursive

# No editable install here: the base image already ran `uv pip install
# --no-deps -e .` at bake time against /home/build/synth-setter (see
# docker/ubuntu22_04/Dockerfile), and devcontainer.json now mounts the
# live workspace on top of that same path — so the existing .pth file
# already points at the workspace.

# Pre-commit hooks (pre-commit itself is in the image's deps). Strip any
# absolute host-path core.hooksPath that may leak from the host .git/config
# (harmless in Codespaces; bites local devcontainer users).
git config --local --unset-all core.hooksPath 2>/dev/null || true
pre-commit install

# plumb writes a native git hook — re-run pre-commit install to chain it.
if command -v plumb >/dev/null 2>&1; then
  plumb init || true
  pre-commit install
fi

echo "Dev container ready. Run 'make test' to verify."
