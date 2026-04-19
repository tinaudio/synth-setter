#!/usr/bin/env bash
# Dev container host-side initialization.
# Runs on the host before the container is created (initializeCommand).
#
# Responsibilities, in order:
#   1. Refuse to open the devcontainer when `.git` is a pointer file — i.e.
#      a linked worktree, a submodule, or a `git clone --separate-git-dir`
#      repository. The pointer's target is outside the workspaceMount, so
#      git operations inside the container fail partway through
#      post-create.sh (safe.directory config, pre-commit install), leaving
#      the container half-configured. The common case is worktrees.
#   2. Ensure .env exists (Docker --env-file requires the file to be present).
#      Done *after* the refusal so an aborted run doesn't leave a stray .env
#      in the workspace.

set -euo pipefail

if [ -f .git ]; then
  first_line=""
  read -r first_line < .git 2>/dev/null || true
  case "$first_line" in
    gitdir:*)
      cat >&2 <<'ERR'
ERROR: This devcontainer cannot be opened from a workspace whose `.git` is a
pointer file. See github isse #593 for details.

Pointer-file `.git` shows up for git worktrees (the common case),
submodules, and `git clone --separate-git-dir` repositories. The pointer
target lives on the host outside the workspaceMount, so git operations
inside the container can't resolve the gitdir.

Supported workflows (both keep `.git` as a real directory and let you work
on a feature branch):

  1. Host-only:    open the host worktree in a host editor, skip the
                   devcontainer entirely.
  2. Devcontainer: open the main repo in the devcontainer, then create a
                   worktree (or switch branches) INSIDE the container.
                   See docs/getting-started.md section 2g.
ERR
      exit 1
      ;;
  esac
fi

[ -f .env ] || touch .env
