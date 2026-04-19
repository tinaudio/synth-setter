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
#
# WHY THE HARDSTOP (see tinaudio/synth-setter#593 for the full tradeoff
# analysis):
# A pointer-file .git contains an absolute host path to the real gitdir
# (e.g. /home/<user>/<repo>/.git/worktrees/<name> for a worktree). The
# workspaceMount binds only the visible workspace, so that path doesn't
# exist inside the container. Without a resolvable gitdir, git operations
# fail mid-provisioning.
#
# Two supported workflows exist for branch-isolated work; both keep .git as
# a real directory inside the container:
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
# the pointer-file check below AND the `git config` / `pre-commit install`
# lines in .devcontainer/post-create.sh. DO NOT commit those edits.

set -euo pipefail

if [ -f .git ]; then
  first_line=""
  read -r first_line < .git 2>/dev/null || true
  case "$first_line" in
    gitdir:*)
      cat >&2 <<'ERR'
ERROR: This devcontainer cannot be opened from a workspace whose `.git` is a
pointer file.

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

Full tradeoff analysis and escape hatch for parallel-branch devcontainers:
tinaudio/synth-setter#593 and .devcontainer/initialize.sh.
ERR
      exit 1
      ;;
  esac
fi

[ -f .env ] || touch .env
