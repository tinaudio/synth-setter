#!/usr/bin/env bash
# Dev container first-run setup for both Codespaces and local devcontainers.
# Runs once after the container is created. The base image
# (tinaudio/synth-setter:devcontainer-tools) already ships all deps, Surge XT,
# xvfb, and rclone — but NOT credentials. The devcontainer configs do not
# forward `.env` automatically; R2 and W&B creds must be provided at
# runtime via Codespaces secrets or other devcontainer environment-variable
# configuration, or sourced manually inside the container shell.
set -euo pipefail

# Install ~/.tmux.conf for the current user (the selectable tmux terminal
# profile in each .devcontainer/*/devcontainer.json auto-discovers this file).
# Done before the root→dev exec below so the root variant's /root/.tmux.conf
# is populated for terminals that open as root.
_devc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install -m 0644 "$_devc_dir/tmux.conf" "$HOME/.tmux.conf"

# Install the zellij config (default VS Code terminal profile) like tmux.conf
# above — silences startup popups, shares one session; pre-exec so root is covered.
install -D -m 0644 "$_devc_dir/zellij.kdl" "$HOME/.config/zellij/config.kdl"

# Make the bundled coding agents non-interactive by default. Seeded before the
# root→dev exec so it covers whichever $HOME is current, like ~/.tmux.conf above.
configure_agent_autonomy() {
  # Codex reads ~/.codex/config.toml on every invocation. Seed full-auto
  # defaults only when absent so a user- or mount-provided config still wins.
  local codex_config="$HOME/.codex/config.toml"
  if [[ ! -f "$codex_config" ]]; then
    mkdir -p "$(dirname "$codex_config")"
    cat >"$codex_config" <<'EOF'
# Full-auto defaults for the devcontainer (the container is the sandbox):
# never pause for approval and allow unsandboxed command execution.
approval_policy = "never"
sandbox_mode = "danger-full-access"
EOF
  fi

  # agy has no persisted skip-permissions setting, so wrap it — a function (not
  # alias) so sourced shells inherit it and `command agy` still bypasses it.
  if ! grep -qsF 'agy()' "$HOME/.bashrc"; then
    cat >>"$HOME/.bashrc" <<'EOF'

# agy full-auto — auto-approve all Antigravity tool permissions by default.
agy() {
  command agy --dangerously-skip-permissions "$@"
}
EOF
  fi
}
configure_agent_autonomy

# Drop to `dev` when invoked as root so workspace mutations (git config
# --local, pre-commit install → .git/hooks/*) don't land root-owned in the
# bind-mounted workspace. Both opt-in DEVCONTAINER_USER=root sessions and
# Codespaces (which runs postCreateCommand as root) hit this path.
#
# Before dropping, scrub core.hooksPath from the scopes only root can reach:
# /etc/gitconfig (--system) and /root/.gitconfig (--global). --global is
# per-user, so a stray entry there shadows .git/hooks/pre-commit for any
# later root shell or root-running agent and makes `pre-commit install`
# refuse with "Cowardly refusing…". These writes land outside the workspace,
# so they don't violate the no-root-owned-files invariant above.
if [ "$(id -u)" -eq 0 ]; then
  for scope in --system --global; do
    git config "$scope" --unset-all core.hooksPath 2>/dev/null || true
  done
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

# Correct workspace ownership before the first .git write below. A root-owned
# host checkout bind-mounts in with every file root-owned, so `dev` can't write
# .git, run `pre-commit install`, or commit. Guard on both the workspace root
# and `.git` so a mixed-ownership tree (e.g. `.git` left root-owned by an
# earlier DEVCONTAINER_USER=root session) still self-heals — `.git` is the
# dir the failing writes target. The recursive chown is skipped only when both
# are already correct (post-create time budget), via the NOPASSWD sudo from
# common-utils. `.git` is a real dir here (initialize.sh refuses pointer-file
# `.git`). Capture owners first so a stat failure aborts under `set -e`.
workspace_owner="$(stat -c %u "$dir")"
gitdir_owner="$(stat -c %u "$dir/.git")"
if [ "$workspace_owner" != "$(id -u)" ] || [ "$gitdir_owner" != "$(id -u)" ]; then
  sudo chown -R "$(id -u):$(id -g)" "$dir"
fi

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

# Pre-commit hooks (pre-commit itself is in the image's deps). Strip
# core.hooksPath from every scope dev can write — --local catches the
# host-bind-mounted .git/config leak, --global catches a stray entry in
# /home/dev/.gitconfig, --worktree catches per-worktree overrides. Without
# this, `pre-commit install` aborts with "Cowardly refusing…" and any
# value set here would silently shadow .git/hooks/pre-commit at commit
# time. --system is handled in the root pre-exec above; dev cannot write
# /etc/gitconfig.
for scope in --global --local --worktree; do
  git config "$scope" --unset-all core.hooksPath 2>/dev/null || true
done
pre-commit install

# Per-worktree venv isolation. The image bakes VIRTUAL_ENV=/venv/main onto
# PATH, so every git worktree shares one editable install — whichever ran
# `uv sync` last owns it. Walk up to the .project-root anchor and activate
# that worktree's ./.venv when present, shadowing /venv/main. The harness
# re-sources ~/.bashrc per shell and keeps cwd, so the active venv tracks the
# worktree. `-qs`: an absent ~/.bashrc is the not-yet-installed case, not an
# error. See #1339.
if ! grep -qs 'Per-worktree venv isolation' "$HOME/.bashrc"; then
  cat >>"$HOME/.bashrc" <<'EOF'

# Per-worktree venv isolation — see .devcontainer/post-create.sh.
# Walk up to .project-root so activation works from any subdir, not just the
# worktree root. The VIRTUAL_ENV guard skips re-activation so re-sourcing
# can't stack PATH.
__ss_root="$PWD"
while [[ "$__ss_root" != "/" && ! -f "$__ss_root/.project-root" ]]; do
  __ss_root="$(dirname "$__ss_root")"
done
if [[ -f "$__ss_root/.venv/bin/activate" && "${VIRTUAL_ENV:-}" != "$__ss_root/.venv" ]]; then
  unset VIRTUAL_ENV
  source "$__ss_root/.venv/bin/activate"
fi
unset __ss_root
EOF
fi

# Project skills into ~/.agents/skills for Gemini/Antigravity discovery;
# best-effort (`|| true` keeps a fresh box without the plugin from failing). See #1561.
"$dir/scripts/dev/link-skills.sh" || true

echo "Dev container ready. Run 'make test-fast' to verify."
