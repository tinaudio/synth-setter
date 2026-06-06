#!/usr/bin/env bash
# =============================================================================
# pre-pr-review-gate.sh — PreToolUse hook, gates `gh pr create`
# =============================================================================
#
# PURPOSE
# -------
# Block `gh pr create` unless the command carries `REVIEW_FULL=<path>` and that
# path points at a sentinel review file whose name encodes a commit SHA that
# is reachable from the PR's branch tip within `REVIEW_MAX_LAG` first-parent
# commits.
#
# WORKTREE AWARENESS
# ------------------
# The mandated workflow runs each branch in its own `git worktree`, but a
# PreToolUse hook evaluates from the SESSION/PRIMARY checkout (always on the
# default branch). So the ref the checks run against is derived from the
# command's `--head <branch>` (also `-H` / `--head=`), NOT the primary `HEAD`:
#   - ancestry + first-parent lag run against the branch tip (local
#     `refs/heads/<branch>`, else `refs/remotes/origin/<branch>`);
#   - a relative REVIEW_FULL path absent from cwd is resolved against that
#     branch's worktree root (from `git worktree list --porcelain`).
# Absent `--head`, behavior is unchanged (HEAD, cwd). If `--head` is given but
# the branch/worktree can't be resolved, the strict HEAD/cwd behavior stands —
# the gate never loosens.
#
# The sentinel format (basename) is::
#
#     repo-review-full-no-comments.<40-char-sha>.md
#
# Filename format is owned by `agent/_shared/review_sentinel.py`; this script
# shells out to it for parsing so the format has exactly one source of truth
# (shared with the `/repo-review-full-no-comments` skill).
#
# Honor-system gate: an agent could fabricate a sentinel named after a real
# recent SHA, but it must already know a valid recent SHA on this branch.
# Filename-SHA + ancestry + first-parent lag are the floor; beyond grepping for
# unresolved `[comment-hygiene:warn|block]` tags and any `[<skill>:block]` tag
# (the sub-gates below), the gate does not inspect file contents or mtime.
#
# CONTRACT — what the command line must carry
#   REVIEW_FULL=<path>
#     - <path> resolves to an existing file (relative to cwd, or to the
#       `--head` branch's worktree root) that is at least 200 bytes (cheap
#       stub-bypass guard — `touch` produces 0 bytes).
#     - The file's basename matches the sentinel pattern and encodes a SHA.
#     - That SHA is an ancestor of the branch tip AND is at most `REVIEW_MAX_LAG`
#       first-parent commits behind it (default 2; override via env). The
#       `--first-parent` mode means merging `main` into the branch counts as
#       one commit, not the hundreds it brings in.
#     - The file lists no unresolved `[comment-hygiene:warn|block]` findings, unless
#       `REVIEW_COMMENT_GATE` is `warn`/`off` (default `block`).
#     - The file lists no unresolved `[<skill>:block]` findings from any skill,
#       unless `REVIEW_BLOCK_GATE` is `warn`/`off` (default `block`).
#
# INPUT (stdin)
#   JSON with .tool_input.command — the Bash command the agent is about to run.
#
# INVOCATION
#   Registered in the agent's settings (e.g. .claude/settings.json) as a
#   PreToolUse hook on Bash. The handler-level `if: "Bash(gh pr create *)"`
#   guard is unreliable for PreToolUse hooks (see PR #1090 history), so this
#   script re-validates the command itself — same defensive pattern
#   doc-drift.sh / pr-review-resolver.sh use. Non-matching commands exit 0
#   silently.
# =============================================================================
set -euo pipefail

export HOOK_NAME="pre-pr-review-gate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091  # _lib.sh resolved at runtime; not staged together
source "${SCRIPT_DIR}/_lib.sh"

readonly SENTINEL_PY="${SCRIPT_DIR}/../_shared/review_sentinel.py"
readonly MIN_REVIEW_BYTES=200
# gitlint ships its CLI in the `gitlint-core` distribution; pin to the rev in
# .pre-commit-config.yaml so a PR title obeys the same conventional-commit rule
# as commit messages. `uvx` fetches it on demand — no install on PATH needed.
readonly GITLINT_PKG="gitlint-core@0.19.1"
REVIEW_MAX_LAG="${REVIEW_MAX_LAG:-2}"
REVIEW_COMMENT_GATE="${REVIEW_COMMENT_GATE:-block}"
REVIEW_BLOCK_GATE="${REVIEW_BLOCK_GATE:-block}"
PR_TITLE_GATE="${PR_TITLE_GATE:-block}"

INPUT=$(cat)
COMMAND=$(jq -r '.tool_input.command // empty' 2>/dev/null <<<"$INPUT" || true)

# Here-string (not `echo |`) so SIGPIPE under pipefail can't fail-open. Allow
# leading whitespace at line start so `  gh pr create` is still gated.
if ! grep -qE '(^[[:space:]]*|[;|&`(][[:space:]]*)gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)' <<<"$COMMAND"; then
  exit 0
fi

# Extract the PR's head branch from `--head <b>` / `-H <b>` / `--head=<b>`.
# shlex-tokenize so a value isn't split on spaces and the trailing
# `# REVIEW_FULL=...` comment is dropped. Empty when the flag is absent.
extract_head_branch() {
  COMMAND="$1" python3 - <<'PY' 2>/dev/null || true
import os
import shlex

try:
    tokens = shlex.split(os.environ["COMMAND"], comments=True)
except ValueError:
    raise SystemExit(0)
for i, tok in enumerate(tokens):
    if tok in ("--head", "-H") and i + 1 < len(tokens):
        print(tokens[i + 1])
        break
    if tok.startswith("--head="):
        print(tok[len("--head=") :])
        break
PY
}

# Resolve a branch tip to a SHA: prefer the local ref, else origin's. Empty
# (return 1) when neither exists — caller keeps the strict HEAD behavior.
resolve_branch_tip() {
  local branch="$1" sha
  sha=$(git rev-parse --verify --quiet "refs/heads/${branch}" 2>/dev/null) \
    || sha=$(git rev-parse --verify --quiet "refs/remotes/origin/${branch}" 2>/dev/null) \
    || return 1
  printf '%s\n' "$sha"
}

# Echo the worktree root checked out on `refs/heads/<branch>`, parsing the
# porcelain records (`worktree <path>` ... `branch refs/heads/<name>`). Empty
# (return 1) when no worktree holds the branch — caller keeps cwd resolution.
# The porcelain is passed via env, not a pipe: the heredoc already owns the
# python process's stdin, so a piped feed would be silently discarded.
worktree_root_for_branch() {
  local branch="$1" porcelain
  porcelain=$(git worktree list --porcelain 2>/dev/null) || return 1
  BRANCH="$branch" PORCELAIN="$porcelain" python3 - <<'PY' 2>/dev/null || return 1
import os

want = "refs/heads/" + os.environ["BRANCH"]
path = None
for line in os.environ["PORCELAIN"].splitlines():
    if line.startswith("worktree "):
        path = line[len("worktree ") :]
    elif line == f"branch {want}" and path is not None:
        print(path)
        break
PY
}

# Resolve the ref the ancestry/lag checks run against and the dir a relative
# REVIEW_FULL path resolves against. With `--head`, derive both from the branch;
# a branch/worktree that can't be resolved leaves the strict HEAD/cwd default,
# so the gate never loosens.
TARGET_BRANCH=$(extract_head_branch "$COMMAND")
REVIEW_REF="HEAD"
REVIEW_REF_LABEL="HEAD"
WORKTREE_ROOT=""
if [[ -n "$TARGET_BRANCH" ]]; then
  if branch_tip=$(resolve_branch_tip "$TARGET_BRANCH"); then
    REVIEW_REF="$branch_tip"
    REVIEW_REF_LABEL="branch ${TARGET_BRANCH} (${branch_tip})"
  fi
  WORKTREE_ROOT=$(worktree_root_for_branch "$TARGET_BRANCH" || true)
fi

# shellcheck disable=SC2016  # intentional: no expansion wanted in the help block
readonly BLOCK_HELP='Run /repo-review-full-no-comments — it writes the rendered report to
.agent-reviews/repo-review-full-no-comments.<HEAD-sha>.md (filename owned by
agent/_shared/review_sentinel.py). Then re-run with the path in the command
— recommended as a trailing comment so other hooks still fire:

  gh pr create --title "..." --body "..."  # REVIEW_FULL=.agent-reviews/repo-review-full-no-comments.<sha>.md

REVIEW_FULL=<value> runs to the next whitespace or shell metachar. Quoted
paths are NOT recognized — the value-character class excludes both single
and double quotes entirely, so REVIEW_FULL="..." reads as missing-REVIEW_FULL.
Pass the path bare.

The encoded SHA must be an ancestor of HEAD and within REVIEW_MAX_LAG
first-parent commits behind it (default 2; override via the REVIEW_MAX_LAG
env var, must be a non-negative integer).'

# shellcheck disable=SC2016  # intentional: no expansion wanted in the help block
readonly TITLE_HELP='The PR title is the squash-merge subject, so it lands on main as a commit
and must be a conventional commit: <type>(<optional-scope>): <description>.
Allowed <type>s are defined in .gitlint. Set PR_TITLE_GATE=off to bypass
(the pr-metadata-gate workflow re-checks the title regardless).'

block() {
  local reason="$1"
  # Optional 2nd arg overrides the help block; defaults to the REVIEW_FULL help.
  local help="${2:-$BLOCK_HELP}"
  log "blocking: $reason"
  printf 'BLOCKED: %s\n\n%s\n' "$reason" "$help" >&2
  exit 2
}

if [[ ! "$REVIEW_MAX_LAG" =~ ^[0-9]+$ ]]; then
  block "REVIEW_MAX_LAG must be a non-negative integer, got: ${REVIEW_MAX_LAG}"
fi

case "$REVIEW_COMMENT_GATE" in
  block | warn | off) ;;
  *) block "REVIEW_COMMENT_GATE must be one of block|warn|off, got: ${REVIEW_COMMENT_GATE}" ;;
esac

case "$REVIEW_BLOCK_GATE" in
  block | warn | off) ;;
  *) block "REVIEW_BLOCK_GATE must be one of block|warn|off, got: ${REVIEW_BLOCK_GATE}" ;;
esac

case "$PR_TITLE_GATE" in
  block | warn | off) ;;
  *) block "PR_TITLE_GATE must be one of block|warn|off, got: ${PR_TITLE_GATE}" "$TITLE_HELP" ;;
esac

# PR title sub-gate — lint the title against .gitlint before the review-file
# checks so a malformed title fails fast. Best-effort: only an inline --title/-t
# value is visible, and a uvx/network failure fails open (the pr-metadata-gate
# workflow is the authority).
if [[ "$PR_TITLE_GATE" != "off" ]]; then
  # shlex-tokenize the command so a quoted title with spaces/colons survives;
  # comments=True drops the trailing `# REVIEW_FULL=...`. Empty when no flag.
  pr_title=$(COMMAND="$COMMAND" python3 - <<'PY' || true
import os
import shlex

try:
    tokens = shlex.split(os.environ["COMMAND"], comments=True)
except ValueError:
    raise SystemExit(0)
for i, tok in enumerate(tokens):
    if tok in ("--title", "-t") and i + 1 < len(tokens):
        print(tokens[i + 1])
        break
    if tok.startswith("--title="):
        print(tok[len("--title=") :])
        break
PY
  )

  if [[ -n "$pr_title" ]]; then
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
    # `&& ... || title_rc=$?` keeps a non-zero gitlint exit from tripping
    # `set -e`; under pipefail the pipeline carries uvx/gitlint's own code.
    title_output=$(printf '%s\n' "$pr_title" \
      | uvx --from "$GITLINT_PKG" gitlint --config "${repo_root}/.gitlint" 2>&1) \
      && title_rc=0 || title_rc=$?
    # Decide on the OUTPUT, not the exit code: gitlint's code is its violation
    # COUNT (2 problems → exit 2), and uvx's own launch failure is also exit 2,
    # so the code alone can't separate "rejected" from "could not run". A real
    # gitlint verdict line is `<lineno>: <RULEID> ...` (e.g. `1: CT1 ...`).
    if [[ "$title_rc" -eq 0 ]]; then
      :
    elif grep -qE '^[0-9]+: [A-Z]+[0-9]+' <<<"$title_output"; then
      if [[ "$PR_TITLE_GATE" == "warn" ]]; then
        log "title-gate warn: ${title_output}"
        printf 'WARNING: PR title is not a conventional commit:\n%s\n' "$title_output" >&2
      else
        log "blocking (title): ${title_output}"
        printf 'BLOCKED: PR title is not a conventional commit:\n%s\n\n%s\n' \
          "$title_output" "$TITLE_HELP" >&2
        exit 2
      fi
    else
      # No verdict line — uvx/gitlint could not run (offline, no writable cache,
      # not installed). Fail open; CI is the authority.
      log "title-gate fail-open: gitlint exit ${title_rc}: ${title_output}"
      printf 'WARNING: could not lint the PR title (gitlint exit %s); skipped — CI enforces it.\n' \
        "$title_rc" >&2
    fi
  fi
fi

if [[ ! -f "$SENTINEL_PY" ]]; then
  block "missing companion helper at ${SENTINEL_PY} (gate cannot parse sentinel filenames without it)"
fi

# Allow only one REVIEW_FULL= occurrence — multiple would silently pick the
# first and an agent juggling two reviews could pass against the wrong one.
# `|| true` traps grep's no-match exit-1; pipefail would otherwise abort the
# script before the empty-path branch below can emit its own BLOCKED message.
review_full_matches=$(grep -oE 'REVIEW_FULL=[^[:space:]#;|&`()<>"'"'"']+' <<<"$COMMAND" || true)
review_full_count=$(printf '%s' "$review_full_matches" | grep -c . || true)
if [[ "$review_full_count" -gt 1 ]]; then
  block "multiple REVIEW_FULL= tokens in command (${review_full_count}); pick exactly one"
fi

# `grep` returns 1 on no match (under pipefail that aborts the script) and
# `head -1`'s SIGPIPE on closing the pipe early would also surface — `|| true`
# traps both so the empty-path branch below can emit a clean BLOCKED message.
REVIEW_PATH=$(grep -oE 'REVIEW_FULL=[^[:space:]#;|&`()<>"'"'"']+' <<<"$COMMAND" | head -1 | cut -d= -f2- || true)

if [[ -z "$REVIEW_PATH" ]]; then
  block "gh pr create is missing REVIEW_FULL=<path-to-review-file>"
fi

# A relative path absent from cwd is resolved against the `--head` branch's
# worktree root, where /repo-review-full-no-comments wrote the sentinel. An
# absolute path is honored as-is. Only rewrite when the worktree-rooted file
# exists, so a genuinely-missing path still reports its original value.
if [[ ! -f "$REVIEW_PATH" && "$REVIEW_PATH" != /* && -n "$WORKTREE_ROOT" \
  && -f "${WORKTREE_ROOT}/${REVIEW_PATH}" ]]; then
  REVIEW_PATH="${WORKTREE_ROOT}/${REVIEW_PATH}"
fi

if [[ ! -f "$REVIEW_PATH" ]]; then
  block "REVIEW_FULL path does not point at a file: $REVIEW_PATH"
fi

# Inode-metadata size read (no file contents touched). GNU `stat -c %s`
# vs BSD/macOS `stat -f %z` — try the GNU form first and fall back; both
# return a bare integer suitable for `-lt`.
if ! review_size=$(stat -c %s "$REVIEW_PATH" 2>/dev/null); then
  review_size=$(stat -f %z "$REVIEW_PATH" 2>/dev/null || echo "")
fi
if [[ -z "$review_size" ]]; then
  block "could not stat REVIEW_FULL path: $REVIEW_PATH"
fi
if [[ "$review_size" -lt "$MIN_REVIEW_BYTES" ]]; then
  block "REVIEW_FULL file is suspiciously small (${review_size} < ${MIN_REVIEW_BYTES} bytes — likely a touch-bypass): $REVIEW_PATH"
fi

# Filename -> SHA via the shared Python helper (single source of truth). The
# helper exits 1 when the basename doesn't match the sentinel pattern, exit 2
# on usage error or ValueError — surface those separately so a missing
# python3 / broken helper isn't misreported as "bad filename".
helper_stderr=$(mktemp)
trap 'rm -f "$helper_stderr"' EXIT
if review_sha=$(python3 "$SENTINEL_PY" parse "$REVIEW_PATH" 2>"$helper_stderr"); then
  :
else
  helper_rc=$?
  helper_msg=$(cat "$helper_stderr")
  case "$helper_rc" in
    1)
      block "REVIEW_FULL filename does not match the sentinel pattern 'repo-review-full-no-comments.<40-char-sha>.md': $REVIEW_PATH"
      ;;
    *)
      block "internal helper error (exit ${helper_rc}) parsing ${REVIEW_PATH}: ${helper_msg:-no stderr captured}"
      ;;
  esac
fi

# Ancestry: a SHA on a sibling branch (or one rewritten by rebase/amend) is
# rejected outright — the review covers a different line of history. The ref is
# the `--head` branch tip when given, else HEAD (see WORKTREE AWARENESS).
if ! git merge-base --is-ancestor "$review_sha" "$REVIEW_REF" 2>/dev/null; then
  block "review SHA ${review_sha} is not an ancestor of ${REVIEW_REF_LABEL} (rebase/amend rewrote history? run /repo-review-full-no-comments again)"
fi

# First-parent lag — merging origin/main counts as one commit, not the
# dozens it brings in.
lag=$(git rev-list "${review_sha}..${REVIEW_REF}" --first-parent --count 2>/dev/null || echo "")
if [[ -z "$lag" ]]; then
  block "could not compute first-parent lag between review SHA ${review_sha} and ${REVIEW_REF_LABEL}"
fi
if [[ "$lag" -gt "$REVIEW_MAX_LAG" ]]; then
  block "review is ${lag} first-parent commits behind ${REVIEW_REF_LABEL} (max ${REVIEW_MAX_LAG}; set REVIEW_MAX_LAG=N to widen)"
fi

# Reject any sentinel still listing findings. Match the bracketed
# `[comment-hygiene:<severity>]` tag, not the bare skill name the PASS template
# uses. `|| true`: tolerate grep's no-match exit-1, like the counter above.
if [[ "$REVIEW_COMMENT_GATE" != "off" ]]; then
  comment_findings=$(grep -oE '\[comment-hygiene:(warn|block)\]' "$REVIEW_PATH" || true)
  comment_count=$(printf '%s' "$comment_findings" | grep -c . || true)
  if [[ "$comment_count" -gt 0 ]]; then
    remediation="run /fix-review-comments, then refresh the sentinel with /repo-review-full-no-comments (REVIEW_COMMENT_GATE=off bypasses for an intentional finding)"
    if [[ "$REVIEW_COMMENT_GATE" == "warn" ]]; then
      log "comment-gate warn: ${comment_count} unresolved comment-hygiene finding(s) in ${REVIEW_PATH}"
      printf 'WARNING: review still lists %s unresolved comment-hygiene finding(s) in %s — %s\n' \
        "$comment_count" "$REVIEW_PATH" "$remediation" >&2
    else
      block "review still lists ${comment_count} unresolved comment-hygiene finding(s) in ${REVIEW_PATH} — ${remediation}"
    fi
  fi
fi

# Reject any sentinel still listing a BLOCK finding from a non-comment-hygiene
# skill (`[<skill>:block]`: synth-setter, code-health, ml-test, pr-health, ...).
# comment-hygiene blocks are the comment sub-gate's domain, excluded here so the
# gates don't overlap and REVIEW_COMMENT_GATE=off fully owns comment-hygiene.
# `|| true`: tolerate grep's no-match exit-1, like the comment sub-gate above.
if [[ "$REVIEW_BLOCK_GATE" != "off" ]]; then
  block_findings=$(grep -oE '\[[a-z][a-z0-9-]*:block\]' "$REVIEW_PATH" \
    | grep -vF '[comment-hygiene:block]' || true)
  block_count=$(printf '%s' "$block_findings" | grep -c . || true)
  if [[ "$block_count" -gt 0 ]]; then
    block_remediation="resolve them or set REVIEW_BLOCK_GATE=off for an intentional override"
    if [[ "$REVIEW_BLOCK_GATE" == "warn" ]]; then
      log "block-gate warn: ${block_count} unresolved BLOCK finding(s) in ${REVIEW_PATH}"
      printf 'WARNING: review still lists %s unresolved BLOCK finding(s) in %s — %s\n' \
        "$block_count" "$REVIEW_PATH" "$block_remediation" >&2
    else
      block "review still lists ${block_count} unresolved BLOCK finding(s) in ${REVIEW_PATH} — ${block_remediation}"
    fi
  fi
fi

log "review accepted: ${REVIEW_PATH} (sha=${review_sha}, ref=${REVIEW_REF_LABEL}, lag=${lag}/${REVIEW_MAX_LAG}, size=${review_size}B)"
exit 0
