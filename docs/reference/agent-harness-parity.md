# Agent harness parity: Claude Code and Codex

synth-setter keeps a single agent contract in [`AGENTS.md`](../../AGENTS.md) and a single set of
shared assets under `agent/` (`hooks/`, `skills/`). `.claude/{hooks,skills}` are symlinks to that
tree, and the Codex CLI discovers the same skills through its plugin manifest. This doc records the
one place the two harnesses genuinely diverge — **hook enforcement** — and the server-side CI gate
that backstops each blocking hook when the client-side block is unavailable.

## Why hooks are the divergence

Claude Code runs `PreToolUse` and `Stop` hooks that can **block** an action before it happens
(exit 2). Codex observes tool use *after the fact* (its `AfterAgent` / `AfterToolUse` model) and
relies on its sandbox + approval policy rather than a pre-execution veto. The hook scripts under
`agent/hooks/` are provider-neutral bash and run identically under either harness; what differs is
whether a non-zero exit *prevents* the action. So a workflow invariant must never depend solely on
a client-side blocking hook — every blocking hook needs a server-side backstop, and that is what
the audit below pins.

The capability tier is expressed through the existing mode env-vars
(`WORKTREE_GUARD_MODE`, `REVIEW_*_GATE`, `PR_READINESS_GATE`: `block` / `warn` / `off`), not a
rewrite: under Codex, a `warn`-mode hook still emits its message post-hoc, and the CI gate is the
real control.

## Blocking-hook audit

| Hook (event)                                       | What it blocks                                                          | Client-side mode                                                                | Server-side CI backstop                                                                                                                                                                                                                                                                                               |
| -------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `worktree-guard.sh` (PreToolUse Edit/Write)        | Edits inside the primary checkout                                       | `WORKTREE_GUARD_MODE` (default `warn`)                                          | None needed — every change reaches `main` only through a PR branch, so a wrong-checkout edit is caught by ordinary git/PR review, not CI. Local-safety only.                                                                                                                                                          |
| `no-baseline-additions.sh` (PreToolUse Edit/Write) | New rows in `.pydoclint-baseline.txt`                                   | Always blocks (no mode knob)                                                    | **Partial.** `code-quality-pr.yaml` runs pydoclint on changed files and the `check_no_new_funcs_in_pydoclint_excluded.py` ratchet (#938), but neither fails on baseline-*row* growth — the baseline is an allowlist that keeps pydoclint green. This hook is the primary control; PR review is the residual backstop. |
| `git-commit-trailer-check.sh` (PreToolUse Bash)    | `Co-Authored-By` / agent-attribution trailers                           | Always blocks (no mode knob)                                                    | **Partial.** No dedicated server-side trailer gate. Squash-merge composes the merge commit from the PR and drops per-commit trailers; `pr-metadata-gate.yaml`'s `check-pr-title` gitlint pass covers the resulting subject line.                                                                                      |
| `no-yaml-run-comments.sh` (PreToolUse Edit/Write)  | Comments inside YAML `run:` / `setup:` block scalars                    | Always blocks (no mode knob)                                                    | **Strong.** A malformed block scalar fails the workflow at execution; `test-act.yaml` parses every workflow (`act -l` / `-n`) and `tests/infra/test_workflows_under_act.py` exercises them on each PR.                                                                                                                |
| `pre-pr-review-gate.sh` (PreToolUse Bash)          | `gh pr create` without a fresh `/repo-review-full-no-comments` sentinel | `REVIEW_COMMENT_GATE` / `REVIEW_BLOCK_GATE` / `PR_TITLE_GATE` (default `block`) | **Strong.** This gate only front-loads findings locally. The real merge gates are `test.yml` (unit suite), `code-quality-pr.yaml` (pre-commit), `pr-metadata-gate.yaml` (linked issue + title), and Copilot review.                                                                                                   |
| `pr-readiness-stop.sh` (Stop)                      | Ending a turn while CI is red or the PR is not `MERGEABLE`              | `PR_READINESS_GATE` (default `block`)                                           | **Strong.** Branch protection + required status checks + Copilot review gate the merge regardless of whether the local Stop hook fired; `warn` mode loses only the turn-level nag, not merge safety.                                                                                                                  |

Reading the table: under Codex (observe-only), the **Strong** rows are fully covered server-side, so
losing the client-side block costs nothing but earlier feedback. The two **Partial** rows
(`no-baseline-additions`, `git-commit-trailer-check`) are the residual risk — there the hook is the
primary control and human PR review is the only backstop. Adding a server-side check for either is
tracked follow-up work, not part of this parity layer (YAGNI until a Codex user actually hits it).

## Skill discovery parity

`agent/hooks/_lib.sh`'s `has_skill` resolves a skill across both harness install layouts:

- Claude marketplace: `~/.claude/plugins/<marketplace>/skills/<name>/SKILL.md`
- Codex plugin manifest: `~/.codex/plugins/<marketplace>/codex/synth-setter-skills/<name>/SKILL.md`
  or `~/.codex/plugins/<marketplace>/skills/<name>/SKILL.md`, and the flat
  `~/.codex/skills/<name>/SKILL.md`

`tests/claude_hooks/test_skill_discovery_parity.py` asserts every shipped `agent/skills/<name>`
resolves through *both* globs, so the two stay symmetric as skills are added. The full bash hook
suite (`agent/hooks/test.sh`) runs in CI via
`tests/infra/test_agent_hooks_suite.py` under a simulated Codex skill layout.
