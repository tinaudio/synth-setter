# AGENTS.md

Canonical agent instructions for synth-setter. Shared by Claude and Codex.

## Project

synth-setter: synth inversion, sound matching, preset exploration tools.
Python 3.10+, PyTorch Lightning, Hydra, distributed data pipeline on
SkyPilot-managed compute (RunPod + OCI), stored in Cloudflare R2.
Architecture: [docs/architecture.md](docs/architecture.md).

## Always

- **Always work in an isolated git worktree.** Branch switching and stash
  conflicts have caused lost work and accidental commits to wrong branches.
  Use `git worktree add` (or `isolation: "worktree"` when spawning subagents).
  The main checkout is read-only — `git log`, exploration, `rclone ls`. Never
  edit files there.
- **Always verify the branch before push.** Run `git branch --show-current`
  and confirm it matches the target PR branch. A hook prints the branch on
  every `git commit`; don't ignore it.
- **Never commit without explicit permission.** The user opts in. Pre-commit
  hooks must not be skipped — see [`### Commits`](#commits).
- **Never run `make docker-*` or RunPod commands without asking.** These
  spend money and burn cluster state.

## Writing code

- Pydantic `BaseModel(strict=True)` at trust boundaries (config parsing, JSON
  from R2, worker reports). Dataclasses for internal typed containers.
- `structlog` in pipeline code; stdlib `logging` elsewhere.
- All `rclone` operations use `--checksum`.
- Run `make format` before committing. Pre-commit (ruff, ruff-format,
  pydoclint, prettier, mdformat, gitlint) is authoritative; suppressing
  rules to make CI green is forbidden — see
  [`### Lint exceptions are append-frozen`](#lint-exceptions-are-append-frozen).

## Testing

- `make test-fast` is the default CPU loop; `@pytest.mark.slow` for slow.
- Test names: `test_<what>_<condition>_<expected>`.
- Mutation testing: [docs/testing/mutmut.md](docs/testing/mutmut.md).

## Design defaults

YAGNI. Start minimal, expand only when asked. Don't introduce a new class,
config schema, or pattern speculatively — ask "do we need this *now*?" and
default to no. Refactoring follows real needs, not anticipated ones. Present
a plan before writing code for any non-trivial change.

## Mandatory skills for code changes

Whenever an agent modifies non-documentation code (anything other than `.md` /
`docs/` edits), invoke in order:

1. `/tdd-implementation` — drive the change test-first.
2. `/code-health` — review and clean up the result.
3. `/simplify` — final reuse and efficiency pass.

Pure docs edits are exempt; no other exemptions.

## Commits

- Conventional commits, gitlint-enforced. `internal-feat:` / `internal-fix:`
  for unreleased code (no version bump).
- Scope is skill-bound — see `/github-taxonomy`.
- **Never `--no-verify` / `-n`.** Pre-commit and gitlint must run. Hooks
  work inside worktrees.
- **Never add `Co-Authored-By` trailers** or agent-attribution footers
  ("Generated with …", "Claude …", etc.).
- A `PreToolUse` hook (`agent/hooks/git-commit-trailer-check.sh`) blocks
  violations; if a hook fails, fix the underlying cause — don't bypass.

## Lint exceptions are append-frozen

`.pydoclint-baseline.txt` (#938), `pyproject.toml`'s
`[tool.ruff.lint.per-file-ignores]` / `[tool.ruff].extend-exclude`,
`.pre-commit-config.yaml` per-hook `exclude:` regexes, and
`pyrightconfig.json`'s `"exclude"` are **append-frozen**. The only allowed
edit is a **removal** via the `/lint-cleanup` workflow (one file per PR,
`chore(lint):` prefix). `[tool.pydoclint].exclude` is infra-only after #1044
and must not be edited at all.

A `PreToolUse` hook (`agent/hooks/no-baseline-additions.sh`) blocks new rows
in `.pydoclint-baseline.txt`. If a check fails on a file your PR touches,
the remediation is to fix the underlying lint — never register the file as
exempt.

## YAML `run:` block scalars are bash

In GitHub Actions workflows (`.github/workflows/*.{yml,yaml}`) and SkyPilot
task configs (`configs/compute/*.yaml`'s `run:` / `setup:` blocks), comments
go **above** the step, never inside the block scalar. The block-scalar body
is bash and stray `'`, `` ` ``, `$`, or `\` inside a comment has caused
unintended shell expansion. A `PreToolUse` hook
(`agent/hooks/no-yaml-run-comments.sh`) enforces this.

## PRs

- **Every PR body links a taxonomy-compliant issue** via `Closes #N`,
  `Fixes #N`, `Refs #N`, or `Part of #N`. Use `Refs #N` for partial fixes
  (`Fixes` auto-closes). Every issue traces to an Epic via Phase → Task /
  Bug / Feature. See `/github-taxonomy`.
- **PR titles stand alone.** Name the specific subject, not just the action:
  reviewers and `git log` readers don't open the issue. `/github-taxonomy`
  has the canonical title rule and examples.
- **Pre-PR review gate.** Before `gh pr create`, run
  `/repo-review-full-no-comments` and address every BLOCK/WARN (fix code or
  document why it's intentional). The skill writes the rendered report to
  `.agent-reviews/repo-review-full-no-comments.<HEAD-sha>.md` — filename
  format owned by `agent/_shared/review_sentinel.py`, shared with the gate
  hook. A `PreToolUse` hook (`agent/hooks/pre-pr-review-gate.sh`) blocks
  `gh pr create` until the command carries `REVIEW_FULL=<path>` pointing at
  that file — recommended as a trailing comment so other gh-pr-create hooks
  still fire:
  `gh pr create … # REVIEW_FULL=.agent-reviews/repo-review-full-no-comments.<sha>.md`.
  The encoded SHA must be an ancestor of HEAD and within `REVIEW_MAX_LAG`
  (default 2) first-parent commits of it — merges from main count as one
  commit, not the dozens they bring in. Set `REVIEW_MAX_LAG=N` for a
  justified larger gap.
- **Readiness gates:** CI green ∧ `mergeable=MERGEABLE` ∧ every review
  comment has an inline reply ∧ no fresh Copilot findings — see
  `/pr-preflight`.
- **After every push, drive the readiness loop until all four gates hold.**
  "I pushed the fix" is not "the PR is ready." Watch CI (`gh pr checks <N> --watch` or `/loop`); on red, diagnose, fix, push, repeat. Reply inline on
  every open review comment via `/pr-review-resolver`. Then wait ~60s (allow
  15 min) for Copilot's post-push review on **both**
  `repos/<OWNER>/<REPO>/pulls/<N>/comments` and
  `repos/<OWNER>/<REPO>/pulls/<N>/reviews`; address any new findings and
  loop. If Copilot is silent past 15 min, manually re-request and repeat at
  most once. Full procedure (commands, endpoints, traps) in
  [`docs/pr-readiness-loop.md`](docs/pr-readiness-loop.md).
- **Always reply inline** on each open PR review comment (humans + Copilot),
  with a fix-commit SHA or justification. Use `/pr-review-resolver`.
- **Verification evidence** for each behavioral claim goes through
  `/pr-checkbox`.
- **In chat**, use full markdown hyperlinks for PR/issue references:
  `[#N](https://github.com/tinaudio/synth-setter/issues/N)`. In PR / issue
  bodies, use bare `Fixes #N` so GitHub auto-close works.

## Code review

Local skills wrap the review workflow:

- `/repo-review` (MVP, single agent, inline checklist).
- `/repo-review-full` (parallel agents, posts inline review comments).
- `/repo-review-full-no-comments` (same fan-out, renders to chat — pre-PR
  gate uses this).

See [`agent/skills/repo-review/SKILL.md`](agent/skills/repo-review/SKILL.md)
and the shared analysis in
[`agent/skills/_shared/repo-review-full-analysis.md`](agent/skills/_shared/repo-review-full-analysis.md).

## Refactoring

When moving or renaming code, grep ALL file types — not just `.py`. Include
`.yaml`/`.yml`, `.md`, `.json`, `.toml`, `.sh`, and `Dockerfile`. Use
`/tdd-refactor`, which exhaustively discovers references and pins the
contract.

## GPU verification

Before claiming "no GPU available", run both probes and paste the output:

```bash
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

Only skip if BOTH report no usable GPU. If they disagree, document it as an
environment/setup mismatch — not "no GPU available".

## Commands

```bash
make test-fast       # CPU-only fast tests
make test-full-cpu   # all CPU tests
make test-full-gpu   # GPU + CPU, serial
make format          # pre-commit hooks
make help            # everything else
```
