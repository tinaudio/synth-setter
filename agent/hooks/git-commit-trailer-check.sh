#!/usr/bin/env bash
# git-commit-trailer-check.sh — PreToolUse Bash gate (`Bash(git commit *)`)
# blocking `git commit` calls that skip hooks (`--no-verify` / `-n`, incl.
# bundled short flags) or embed `Co-Authored-By` / agent-attribution trailers
# (AGENTS.md). Reads tool-call JSON on stdin, inspects `-m`/`-F` message text;
# exits 0 (clean / not `git commit`) or 2 (offending flag or trailer found).
set -euo pipefail

export HOOK_NAME="git-commit-trailer-check"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
# Fail closed: blocking malformed input is the whole point of this gate.
if ! COMMAND=$(jq -r '.tool_input.command // empty' <<<"$INPUT" 2>/dev/null); then
  log "jq parse failed; blocking conservatively"
  echo "BLOCKED: git-commit-trailer-check could not parse tool-call JSON." >&2
  exit 2
fi

# Defensive re-scope: handler-level `if:` already filters, but re-validate
# in-script to stay safe if the matcher fires more broadly.
if ! grep -qE '(^[[:space:]]*|[;|&`(][[:space:]]*)git[[:space:]]+commit([[:space:]]|$)' <<<"$COMMAND"; then
  exit 0
fi

# Python so the parser can argv-slice (so `-n` to `git commit` isn't confused
# with a downstream `grep -n`).
FINDINGS=$(HOOK_COMMAND="$COMMAND" python3 - <<'PY'
import os, shlex, pathlib, re

cmd = os.environ["HOOK_COMMAND"]
try:
    lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens = list(lexer)
except ValueError:
    # Unbalanced quotes; fall back to a literal scan. False positives are
    # acceptable — trailer-shaped substrings rarely appear outside messages.
    # Still split shell control operators so downstream argv slicing does not
    # mis-attribute flags across `&&` / `||` / `;` / `|` / `&`.
    tokens = [
        token
        for token in re.split(r"(\&\&|\|\||[;|&])", cmd)
        if token and not token.isspace()
    ]


def commit_arg_slices(tokens):
    """Yield each [start, end) slice of tokens that is `git commit`'s argv.

    A slice ends at a shell metachar token (``&&`` / ``||`` / ``;`` / ``|``) or
    at end-of-tokens, so a downstream ``grep -n`` after ``&&`` is not mistaken
    for ``git commit -n``.
    """
    metachars = {"&&", "||", ";", "|", "&"}
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == "git" and tokens[i + 1] == "commit":
            start = i + 2
            j = start
            while j < len(tokens) and tokens[j] not in metachars:
                j += 1
            yield start, j
            i = j
        else:
            i += 1


def iter_commit_argvs():
    for start, end in commit_arg_slices(tokens):
        yield tokens[start:end]


findings = []


def has_no_verify_short_flag(argv):
    """Return True if argv has `-n` or any bundled short flag containing `n`.

    Short flags cluster: `git commit -nm "msg"` tokenizes as `["-nm", "msg"]`,
    so a bare ``"-n" in argv`` misses it. Stop scanning at the ``--`` end-of-
    options marker, and skip long flags (``--no-foo``) and positional args.
    """
    for tok in argv:
        if tok == "--":
            return False
        if tok.startswith("--"):
            continue
        if len(tok) >= 2 and tok[0] == "-" and "n" in tok[1:] and tok[1:].isalpha():
            return True
    return False


# --no-verify / -n (incl. bundled `-nm` / `-anm` / …) on `git commit` itself.
for argv in iter_commit_argvs():
    if "--no-verify" in argv:
        findings.append(("--no-verify flag", "--no-verify"))
        break
    if has_no_verify_short_flag(argv):
        findings.append(("-n flag (== --no-verify)", "-n"))
        break

# Forbidden trailers inside `-m` / `-F` / heredoc bodies. `Co-Authored-By:` and
# `Generated with` are bare-substring matches — the literal strings rarely
# appear outside attribution contexts and we must match both heredoc-style
# real newlines AND the literal `\n` chars shlex produces from
# `-m "feat: x\n\nCo-Authored-By: …"`. The Claude-model regex IS anchored to
# trailer context so a subject like "feat: tokeniser for Claude Sonnet 4.5"
# does not trigger.
trailer_patterns = [
    ("Co-Authored-By trailer", re.compile(r"Co-Authored-By:", re.IGNORECASE)),
    ("agent-attribution footer", re.compile(r"Generated with", re.IGNORECASE)),
    ("agent-attribution footer", re.compile(r"(?:^|\\n|\n)\s*[A-Za-z-]+:\s*Claude\s+(Code|Opus|Sonnet|Haiku)\b")),
    ("agent-attribution footer", re.compile(r"\bnoreply@anthropic\.com\b")),
]


def collect_message_texts(argv):
    """Extract every commit-message body from one `git commit` argv slice."""
    texts = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-m", "--message") and i + 1 < len(argv):
            texts.append(argv[i + 1])
            i += 2
            continue
        if tok.startswith("--message="):
            texts.append(tok.split("=", 1)[1])
            i += 1
            continue
        if tok in ("-F", "--file") and i + 1 < len(argv):
            try:
                texts.append(pathlib.Path(argv[i + 1]).read_text())
            except OSError:
                pass
            i += 2
            continue
        if tok.startswith("--file="):
            try:
                texts.append(pathlib.Path(tok.split("=", 1)[1]).read_text())
            except OSError:
                pass
            i += 1
            continue
        i += 1
    return texts


texts = []
for argv in iter_commit_argvs():
    texts.extend(collect_message_texts(argv))
if not texts:
    # Heredoc / shell-substituted body: scan the raw command text instead.
    texts = [cmd]

for text in texts:
    for label, pattern in trailer_patterns:
        m = pattern.search(text)
        if m:
            findings.append((label, m.group(0)))
            break

seen = set()
for label, match in findings:
    key = (label, match)
    if key in seen:
        continue
    seen.add(key)
    print(f"{label}\t{match}")
PY
)

if [[ -n "$FINDINGS" ]]; then
  log "blocking forbidden flag/trailer"
  {
    echo "BLOCKED: \`git commit\` invocation is non-compliant."
    echo
    echo "Findings:"
    printf '  %s\n' "${FINDINGS//$'\n'/$'\n  '}"
    echo
    echo "Rules (AGENTS.md):"
    echo "  - Never use --no-verify / -n; hooks work inside worktrees and must run."
    echo "  - Never add Co-Authored-By trailers."
    echo "  - Never add agent-attribution footers (\"Generated with ...\", \"Claude ...\", etc.)."
    echo
    echo "Fix the underlying hook failure (or rewrite the commit message) and re-run."
  } >&2
  exit 2
fi

exit 0
