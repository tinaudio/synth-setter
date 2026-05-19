#!/usr/bin/env bash
# Dev container first-run setup for both Codespaces and local devcontainers.
# Runs once after the container is created. The base image
# (tinaudio/synth-setter:devcontainer-tools) already ships all deps, Surge XT,
# xvfb, and rclone — but NOT credentials. The devcontainer configs do not
# forward `.env` automatically; R2 and W&B creds must be provided at
# runtime via Codespaces secrets or other devcontainer environment-variable
# configuration, or sourced manually inside the container shell.
set -euo pipefail

# Install ~/.tmux.conf for the current user (the VS Code terminal profile in
# each .devcontainer/*/devcontainer.json launches tmux, which auto-discovers
# this file). Done before the root→dev exec below so the root variant's
# /root/.tmux.conf is populated for terminals that open as root.
_devc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install -m 0644 "$_devc_dir/tmux.conf" "$HOME/.tmux.conf"

# TPM (tmux plugin manager) bootstrap + plugin install. tmux.conf declares
# tmux-resurrect/tmux-continuum via `set -g @plugin` lines; TPM's run line at
# the bottom of tmux.conf loads them at tmux start. Plugins live per-$HOME, so
# this runs once for root and again after the runuser exec below for `dev`.
_tpm_dir="$HOME/.tmux/plugins/tpm"
if [ -d "$_tpm_dir/.git" ]; then
  git -C "$_tpm_dir" pull --ff-only --quiet \
    || echo "WARNING: failed to update TPM in $_tpm_dir; using existing checkout." >&2
else
  git clone --quiet --depth 1 https://github.com/tmux-plugins/tpm "$_tpm_dir" \
    || echo "WARNING: failed to clone TPM into $_tpm_dir; tmux plugins won't load until re-run." >&2
fi

# tmux-resurrect state dir — devcontainer.json mounts a named volume here so
# saved sessions survive container rebuilds. mkdir is a no-op when mounted.
mkdir -p "$HOME/.local/share/tmux/resurrect"

# Non-interactive plugin install equivalent to hitting prefix+I inside tmux.
# Tolerate failure (e.g. egress blocked); a warning is enough since users can
# always run prefix+I later from inside tmux.
if [ -x "$_tpm_dir/bin/install_plugins" ]; then
  "$_tpm_dir/bin/install_plugins" >/dev/null 2>&1 \
    || echo "WARNING: TPM install_plugins failed; run prefix+I inside tmux after start." >&2
fi

# Drop to `dev` when invoked as root so workspace mutations (git config
# --local, pre-commit install → .git/hooks/*) don't land root-owned in the
# bind-mounted workspace. Both opt-in DEVCONTAINER_USER=root sessions and
# Codespaces (which runs postCreateCommand as root) hit this path.
if [ "$(id -u)" -eq 0 ]; then
  exec runuser -u dev -- bash "$(readlink -f "${BASH_SOURCE[0]}")"
fi

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

echo "Dev container ready. Run 'make test-fast' to verify."
