# CLAUDE.md

synth-setter: synth inversion, sound matching, and preset-exploration tools — Python 3.10+, PyTorch Lightning, Hydra, with a distributed data pipeline on SkyPilot-managed compute (RunPod + OCI) stored in Cloudflare R2.

Shared agent instructions for Claude and Codex; AGENTS.md is the canonical source. Architecture: [docs/architecture.md](docs/architecture.md).

<important if="you are about to edit, write, or commit any file">

- **Work in an isolated git worktree, never the primary checkout.** Branch switching and stash conflicts have lost work and committed to wrong branches. Use `git worktree add` (or `isolation: "worktree"` when spawning subagents). The primary checkout is read-only — `git log`, exploration, `rclone ls` only. A `SessionStart` banner and a `PreToolUse` guard (`agent/hooks/worktree-guard.sh`, `WORKTREE_GUARD_MODE`: `warn` default / `block` / `off`) enforce this.
- **Each worktree gets its own `.venv`.** The spawn command runs `uv sync`; `~/.bashrc` then activates `./.venv` per directory, overriding the image's shared `/venv/main`. For one-offs, `uv run <cmd>` targets the worktree env regardless of the inherited `VIRTUAL_ENV`.
  </important>

<important if="you need to run commands to build, test, lint, or format">

| Command              | What it does                           |
| -------------------- | -------------------------------------- |
| `make test-fast`     | CPU-only fast tests (the default loop) |
| `make test-full-cpu` | All CPU tests                          |
| `make test-full-gpu` | GPU + CPU, serial                      |
| `make format`        | Run the pre-commit hooks               |
| `make help`          | Everything else                        |

Never run `make docker-*` or RunPod commands without asking — they spend money and burn cluster state.
</important>

<important if="you are writing or modifying Python code">

- Pydantic `BaseModel(strict=True)` at trust boundaries (config parsing, JSON from R2, worker reports); dataclasses for internal typed containers.
- `structlog` in pipeline code; stdlib `logging` elsewhere.
- All `rclone` operations use `--checksum`.
  </important>

<important if="you are starting any non-trivial change">

YAGNI. Start minimal and expand only when asked — don't add a class, config schema, or pattern speculatively. Ask "do we need this *now*?" and default to no. Present a plan before writing code.
</important>

<important if="you are modifying non-documentation code (anything beyond .md / docs/ edits)">

Invoke in order: `/tdd-implementation` (drive it test-first) → `/code-health` (review and clean up) → `/simplify` (final reuse and efficiency pass). Pure docs edits are exempt; no other exemptions.
</important>

<important if="you are writing or running tests">

- Test names: `test_<what>_<condition>_<expected>`.
- `@pytest.mark.slow` marks slow tests.
- Mutation testing: [docs/testing/mutmut.md](docs/testing/mutmut.md).
  </important>

<important if="you are committing">

- Conventional commits, gitlint-enforced. `internal-feat:` / `internal-fix:` for unreleased code (no version bump). Scope is skill-bound — see `/github-taxonomy`.
- Run `make format` first; pre-commit (ruff, pydoclint, prettier, mdformat, gitlint) is authoritative. **Never `--no-verify` / `-n`**, and never suppress a rule to make CI green — fix the underlying cause.
- **Never add `Co-Authored-By` or agent-attribution trailers** ("Generated with …", "Claude …"). A `PreToolUse` hook (`agent/hooks/git-commit-trailer-check.sh`) blocks them.
- **Verify the branch before push:** `git branch --show-current` must match the target PR branch.
- **Never commit without explicit permission.** The user opts in.
  </important>

<important if="a lint, pydoclint, or pyright check fails on a file your change touches">

`.pydoclint-baseline.txt` (#938), `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` / `[tool.ruff].extend-exclude`, `.pre-commit-config.yaml` per-hook `exclude:` regexes, and `pyrightconfig.json`'s `"exclude"` are **append-frozen**. The only allowed edit is a **removal** via `/lint-cleanup` (one file per PR, `chore(lint):` prefix); `[tool.pydoclint].exclude` must not be edited at all. Fix the underlying lint — never register a file as exempt. A `PreToolUse` hook (`agent/hooks/no-baseline-additions.sh`) blocks new baseline rows.
</important>

<important if="you are editing GitHub Actions workflows (.github/workflows/*.yml) or SkyPilot compute configs (src/synth_setter/configs/compute/*.yaml)">

Put comments **above** the step, never inside a `run:` / `setup:` block scalar — the body is bash, and a stray `'`, `` ` ``, `$`, or `\` in a comment has caused unintended shell expansion. A `PreToolUse` hook (`agent/hooks/no-yaml-run-comments.sh`) enforces this.
</important>

<important if="you are moving, renaming, or restructuring code">

Grep ALL file types, not just `.py` — include `.yaml`/`.yml`, `.md`, `.json`, `.toml`, `.sh`, and `Dockerfile`. Use `/tdd-refactor`, which exhaustively discovers references and pins the contract.
</important>

<important if="you are opening or driving a pull request">

- **Link a taxonomy-compliant issue** in the body via `Closes #N` / `Fixes #N` / `Refs #N` / `Part of #N` (use `Refs` for partial fixes; `Fixes` auto-closes). Every issue traces to an Epic via Phase → Task / Bug / Feature. See `/github-taxonomy`.
- **PR titles stand alone** — name the specific subject, not just the action; readers don't open the issue.
- **Pre-PR gate:** run `/repo-review-full-no-comments` and address every BLOCK/WARN. A `PreToolUse` hook (`agent/hooks/pre-pr-review-gate.sh`) blocks `gh pr create` until the command carries `REVIEW_FULL=<path>` pointing at the rendered report — recommended as a trailing comment. The encoded SHA must be within `REVIEW_MAX_LAG` (default 2) first-parent commits of HEAD.
- **After every push, drive `/pr-readiness` until all four gates hold:** CI green ∧ `mergeable=MERGEABLE` ∧ every review comment has an inline reply ∧ no fresh Copilot findings. Full procedure: [docs/pr-readiness-loop.md](docs/pr-readiness-loop.md). A `Stop` hook (`agent/hooks/pr-readiness-stop.sh`, `PR_READINESS_GATE`: `block` default / `warn` / `off`) blocks ending the turn while gates 1-2 fail.
- **Reply inline on every open review comment** (humans + Copilot) with a fix-commit SHA or justification, via `/pr-review-resolver`. Verification evidence goes through `/pr-checkbox`.
- **Advisory rewakes carry an origin-HEAD stamp** — compare the `<sha7>` in a `pr-review-resolver` / `doc-drift` rewake to `git rev-parse HEAD`. If they differ the advisory crossed sessions: read it for context, but don't treat it as current-PR work.
- **In chat**, use full markdown links for refs (`[#N](https://github.com/tinaudio/synth-setter/issues/N)`); in PR / issue bodies use bare `Fixes #N` so auto-close works.
  </important>

<important if="you are reviewing code">

- `/repo-review` — MVP, single agent, inline checklist.
- `/repo-review-full` — parallel agents, posts inline review comments.
- `/repo-review-full-no-comments` — same fan-out, renders to chat (the pre-PR gate uses this).

See [agent/skills/repo-review/SKILL.md](agent/skills/repo-review/SKILL.md) and [agent/skills/\_shared/repo-review-full-analysis.md](agent/skills/_shared/repo-review-full-analysis.md).
</important>

<important if="you are about to claim no GPU is available">

Run both probes and paste the output; only skip if BOTH report no usable GPU. If they disagree, document it as an environment/setup mismatch — not "no GPU available".

```bash
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

</important>
