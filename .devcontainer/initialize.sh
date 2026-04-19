#!/usr/bin/env bash
# Dev container host-side initialization.
# Runs on the host before the container is created (initializeCommand).
#
# Responsibilities:
#   1. Ensure .env exists (Docker --env-file requires the file to be present).
#   2. Refuse to open the devcontainer from a git worktree.
#
# WHY THE WORKTREE HARDSTOP (see tinaudio/synth-setter#593 for the full
# tradeoff analysis):
# A linked worktree's .git is a pointer file containing an absolute host path
# into the parent repo's admin directory (e.g. /home/<user>/<repo>/.git/worktrees/<name>).
# The workspaceMount binds only the worktree itself, so that path doesn't exist
# inside the container. Without a usable .git, git operations in post-create.sh
# (git config --global --add safe.directory, pre-commit install) fail partway,
# leaving the container half-configured.
#
# Two supported workflows exist; both let you work on a feature branch:
#   1. Host-only:    open the host worktree in a host editor, skip the
#                    devcontainer entirely.
#   2. Devcontainer: open the main repo in the devcontainer, then create a
#                    worktree (or switch branches) INSIDE the container. The
#                    in-container .git is a directory that resolves normally.
#                    See docs/getting-started.md §2g for the recommended flow.
#
# EDGE CASE — parallel devcontainers for parallel branches (see #593):
# If you genuinely need to run two devcontainers on two worktrees simultaneously
# (e.g. to A/B-test behavior between branches without rebuilding), comment out
# the worktree check below AND the `git config` / `pre-commit install` lines in
# .devcontainer/post-create.sh. DO NOT commit those edits.

set -euo pipefail

[ -f .env ] || touch .env

if [ -f .git ]; then
  first_line=""
  read -r first_line < .git 2>/dev/null || true
  case "$first_line" in
    gitdir:*)
      cat >&2 <<'ERR'
ERROR: This devcontainer cannot be opened from a git worktree.

The workspace's .git is a pointer file referencing a host path that is not
mounted into the container. Supported workflows (both let you work on a
feature branch):

  1. Host-only:    open the host worktree in a host editor, skip the
                   devcontainer entirely.
  2. Devcontainer: open the main repo in the devcontainer, then create a
                   worktree (or switch branches) INSIDE the container.
                   See docs/getting-started.md section 2g.

Full tradeoff analysis and escape hatch for parallel-branch devcontainers:
tinaudio/synth-setter#593 and .devcontainer/initialize.sh.
ERR
      exit 1
      ;;
  esac
fi
