# CLAUDE.md

synth-setter: synth inversion, sound matching, and preset-exploration tools — Python 3.10+, PyTorch Lightning, Hydra, with a distributed data pipeline on SkyPilot-managed compute (RunPod + OCI) stored in Cloudflare R2.

Shared agent instructions for Claude and Codex; AGENTS.md is the canonical source. Architecture: [docs/architecture.md](docs/architecture.md).

<important if="you are about to edit, write, or commit any file">

- **Always work in an isolated git worktree.** Branch switching and stash
  conflicts have caused lost work and accidental commits to wrong branches.
  Use `git worktree add` (or `isolation: "worktree"` when spawning subagents).
  The primary checkout is read-only — `git log`, exploration, `rclone ls`.
  Never edit files there. A `SessionStart` hook
  (`agent/hooks/session-start-cwd-banner.sh`) prints a primary-vs-worktree
  banner on startup/resume/clear/compact (and warns when `.claude/{skills,hooks}`
  symlinks didn't materialize, e.g. `core.symlinks=false`); a `PreToolUse` hook
  (`agent/hooks/worktree-guard.sh`) warns on Edit/Write inside the primary
  checkout (`WORKTREE_GUARD_MODE`: `warn` default / `block` / `off`); a
  `PostToolUse` hook (`agent/hooks/worktree-post-setup.sh`) automatically
  runs `make link-plugins && make link-thoughts` in every new worktree after
  `git worktree add` (fail-safe, exits 0 on any error — see #1343).
- **Each worktree gets its own `.venv`.** The spawn command runs `uv sync`;
  `~/.bashrc` (installed by `.devcontainer/post-create.sh`) then activates
  `./.venv` per directory, overriding the image's shared `/venv/main`. For
  one-offs, `uv run <cmd>` targets the worktree env regardless of the
  inherited `VIRTUAL_ENV`.
- **Always verify the branch before push.** Run `git branch --show-current`
  and confirm it matches the target PR branch. A hook prints the branch on
  every `git commit`; don't ignore it.
- **Pre-commit hooks must not be skipped** — see [`### Commits`](#commits).
- **Never run `make docker-*` or RunPod commands without asking.** These
  spend money and burn cluster state.
  </important>

<important if="you need to run commands to build, test, lint, or format">

```bash
make test-fast       # CPU-only fast tests
make test-full-cpu   # all CPU tests
make test-full-gpu   # GPU + CPU, serial
make format          # pre-commit hooks
make help            # everything else
```

</important>

<important if="you are writing or modifying Python code">

- Pydantic `BaseModel(strict=True)` at trust boundaries (config parsing, JSON
  from R2, worker reports). Dataclasses for internal typed containers.
- `structlog` in pipeline code; stdlib `logging` elsewhere.
- All `rclone` operations use `--checksum`.
- Add an import in the same edit as its first use, or add imports last —
  ruff's `F401` autofix deletes an import that is momentarily unused if
  `make format` runs before the using code lands, costing a re-add cycle.
- Run `make format` before committing. Pre-commit (ruff, ruff-format,
  pydoclint, prettier, mdformat, gitlint) is authoritative; suppressing
  rules to make CI green is forbidden — see
  [`### Lint exceptions are append-frozen`](#lint-exceptions-are-append-frozen).
  </important>

<important if="you are writing inline comments or docstrings">

Code says **what**; comments say **why** — add prose only when it carries a
constraint, unit, semantic, or rationale that names and types can't, and
describe only current behavior (history belongs in the commit message). Keep
comments to one line (cap two), open docstrings with the contract, and supply
`:param:` / `:returns:` / `:raises:` semantics wherever pydoclint expects them —
full rules in the `comment-hygiene` skill.
</important>

<important if="you are starting any non-trivial change">

YAGNI. Start minimal, expand only when asked. Don't introduce a new class,
config schema, or pattern speculatively — ask "do we need this *now*?" and
default to no. Refactoring follows real needs, not anticipated ones. Present
a plan before writing code for any non-trivial change.
</important>

<important if="you are modifying non-documentation code (anything other than .md / docs/ edits)">

Whenever an agent modifies non-documentation code (anything other than `.md` /
`docs/` edits), invoke in order:

1. `/tdd-implementation` — drive the change test-first.
2. `/code-health` — review and clean up the result.
3. `/simplify` — final reuse and efficiency pass.

Pure docs edits are exempt; no other exemptions.
</important>

<important if="you are writing or running tests">

- `make test-fast` is the default CPU loop; `@pytest.mark.slow` for slow.
- Test names: `test_<what>_<condition>_<expected>`.
- Mutation testing: [docs/testing/mutmut.md](docs/testing/mutmut.md).
  </important>

<important if="you are committing">

- Conventional commits, gitlint-enforced. `internal-feat:` / `internal-fix:`
  for unreleased code (no version bump).
- Scope is skill-bound — see `/github-taxonomy`.
- **Never `--no-verify` / `-n`.** Pre-commit and gitlint must run. Hooks
  work inside worktrees.
- **Never add `Co-Authored-By` trailers** or agent-attribution footers
  ("Generated with …", "Claude …", etc.).
- A `PreToolUse` hook (`agent/hooks/git-commit-trailer-check.sh`) blocks
  violations; if a hook fails, fix the underlying cause — don't bypass.
  </important>

<important if="a lint, pydoclint, or pyright check fails on a file your change touches">

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
</important>

<important if="you are editing GitHub Actions workflows (.github/workflows/*.yml) or SkyPilot compute configs (src/synth_setter/configs/compute/*.yaml)">

In GitHub Actions workflows (`.github/workflows/*.{yml,yaml}`) and SkyPilot
task configs (`src/synth_setter/configs/compute/*.yaml`'s `run:` / `setup:` blocks), comments
go **above** the step, never inside the block scalar. The block-scalar body
is bash and stray `'`, `` ` ``, `$`, or `\` inside a comment has caused
unintended shell expansion. A `PreToolUse` hook
(`agent/hooks/no-yaml-run-comments.sh`) enforces this.
</important>

<important if="you are moving, renaming, or restructuring code">

When moving or renaming code, grep ALL file types — not just `.py`. Include
`.yaml`/`.yml`, `.md`, `.json`, `.toml`, `.sh`, and `Dockerfile`. Use
`/tdd-refactor`, which exhaustively discovers references and pins the
contract.
</important>

<important if="you are opening or driving a pull request">

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
  justified larger gap. The gate also **blocks while the sentinel still lists
  `[comment-hygiene:warn|block]` findings** (`REVIEW_COMMENT_GATE`: `block` default /
  `warn` / `off`) and **while it lists any `[<skill>:block]` finding**
  (`REVIEW_BLOCK_GATE`: `block` default / `warn` / `off`). For comment-hygiene
  findings, run `/fix-review-comments` to apply the rewrites, commit, and
  re-review in one pass; other `[<skill>:block]` findings need the underlying
  issue fixed and `/repo-review-full-no-comments` re-run to regenerate the
  sentinel. Set `REVIEW_COMMENT_GATE=off` / `REVIEW_BLOCK_GATE=off` only for a
  finding you've judged intentional. The gate also **blocks while the PR's
  inline `--title` is not a conventional commit** (`PR_TITLE_GATE`: `block`
  default / `warn` / `off`) — best-effort and fails open on any uvx/network
  error, since the `pr-metadata-gate` workflow re-checks the title regardless.
- **Readiness gates:** CI green ∧ `mergeable=MERGEABLE` ∧ every review
  comment has an inline reply ∧ no fresh Copilot findings — see
  `/pr-preflight`.
- **After every push, drive the readiness loop until all four gates hold.**
  "I pushed the fix" is not "the PR is ready." Run `/pr-readiness` to drive the
  loop: watch CI (`gh pr checks <N> --watch` or `/loop`) and fix red; confirm
  `mergeable=MERGEABLE`; reply inline on every open review comment via
  `/pr-review-resolver`; then wait ~60s (allow 15 min) for Copilot's post-push
  review on **both** `repos/<OWNER>/<REPO>/pulls/<N>/comments` and
  `repos/<OWNER>/<REPO>/pulls/<N>/reviews`; address any new findings and loop.
  If Copilot is silent past 15 min, manually re-request and repeat at most
  once. Full procedure (commands, endpoints, traps) in
  [`docs/pr-readiness-loop.md`](docs/pr-readiness-loop.md). A `Stop` hook
  (`agent/hooks/pr-readiness-stop.sh`) enforces this: it blocks ending the turn
  while gates 1-2 (CI green, `mergeable`) fail for the branch's open PR, and
  points back here for gates 3-4 (`PR_READINESS_GATE`: `block` default /
  `warn` / `off`).
- **Always reply inline** on each open PR review comment (humans + Copilot),
  with a fix-commit SHA or justification. Use `/pr-review-resolver`.
- **Advisory rewakes carry an origin-HEAD stamp.** The `pr-review-resolver`
  and `doc-drift` PostToolUse hooks run their headless agents in detached
  worktrees and re-enter the session via `asyncRewake` with a line like
  `pr-review-resolver report for PR #N (branch X, origin HEAD <sha7>) at <path>`. Before acting on one, compare `<sha7>` to the first 7
  characters of `git rev-parse HEAD` (the advisory is the 7-char prefix).
  If they differ the advisory crossed sessions (it was queued by a prior
  agent's push/PR-create that finished after that session ended) — read
  the report for context, but do not treat it as work for the current PR.
- **Verification evidence** for each behavioral claim goes through
  `/pr-checkbox`.
- **In chat**, use full markdown hyperlinks for PR/issue references:
  `[#N](https://github.com/tinaudio/synth-setter/issues/N)`. In PR / issue
  bodies, use bare `Fixes #N` so GitHub auto-close works.
  </important>

<important if="you are reviewing code">

Local skills wrap the review workflow:

- `/repo-review` (MVP, single agent, inline checklist).
- `/repo-review-full` (parallel agents, posts inline review comments).
- `/repo-review-full-no-comments` (same fan-out, renders to chat — pre-PR
  gate uses this).
- `/fix-review-comments` (applies the sentinel's comment-hygiene findings,
  commits, and re-reviews — the remediation half of the pre-PR comment gate).

See [`agent/skills/repo-review/SKILL.md`](agent/skills/repo-review/SKILL.md)
and the shared analysis in
[`agent/skills/_shared/repo-review-full-analysis.md`](agent/skills/_shared/repo-review-full-analysis.md).
</important>

<important if="you are about to claim no GPU is available">

Before claiming "no GPU available", run both probes and paste the output:

```bash
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

Only skip if BOTH report no usable GPU. If they disagree, document it as an
environment/setup mismatch — not "no GPU available".
</important>

<important if="you are about to claim no VST or R2 is available">

This devcontainer ships `SYNTH_SETTER_PLUGIN_PATH` (Surge XT VST) and
`RCLONE_CONFIG_R2_*` (Cloudflare R2 credentials). Before labelling a
`@pytest.mark.requires_vst` or `@pytest.mark.integration_r2` test as
unrunnable, probe:

```bash
ls "${SYNTH_SETTER_PLUGIN_PATH:-plugins/Surge XT.vst3}"
rclone lsd r2:
```

Both succeed in this devcontainer. **Run — do not skip — these tests.**
`make test-vst-cpu` covers `requires_vst`; `uv run pytest -m "integration_r2" -v`
covers R2 e2e. In PR verification tables, list actual pass/fail results rather
than "SKIP: requires VST / R2".
</important>
