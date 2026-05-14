# Lint Cleanup Agent

## Goal

Fix pre-existing lint violations in legacy files one at a time, removing them from the pre-commit exclusion lists in `.pre-commit-config.yaml`. Tracked in #25.

## How this is invoked

This runbook is the canonical workflow. Three entry points delegate to it; edits to the steps below land here and reach every entry point automatically.

| Tool                       | Entry point                        | How to invoke                                                                |
| -------------------------- | ---------------------------------- | ---------------------------------------------------------------------------- |
| Claude Code (programmatic) | `.claude/agents/lint-cleanup.md`   | `Agent(subagent_type: "lint-cleanup", isolation: "worktree", prompt: "...")` |
| Claude Code (interactive)  | `.claude/commands/lint-cleanup.md` | Type `/lint-cleanup <path>` in a Claude Code session                         |
| Copilot Coding Agent       | `.github/copilot-instructions.md`  | Assign a #25 sub-issue to Copilot in the GitHub UI                           |

The entry-point stubs may surface a short cross-reference of the rules most often forgotten (commit prefix, `Refs #25`, isolated-worktree requirement) so a contributor seeing only the stub still gets the load-bearing constraints. They must not paraphrase or fork the workflow steps themselves — those live here, single-sourced.

## Scope

Only formatting, docstrings, and lint fixes. **No functional changes.**

## Picking the next file

When the agent has discretion over which file to work next — i.e. invoked
without a specific target path — **work the exclusion lists in reverse order
of when each entry was added: most recently added first, oldest legacy
entries last (LIFO).**

To rank candidates, run `git blame` against the exclusion-list lines in
`.pre-commit-config.yaml` and `pyproject.toml` and sort by the commit
timestamp that introduced each entry. Skip entries that are already in
flight (open `chore/lint-cleanup/*` PR or a ticked box on
[#25](https://github.com/tinaudio/synth-setter/issues/25)). If two entries
share the same introducing commit, break the tie with the smaller file by
line count.

**Why:** a freshly-added exclusion almost always came from a small,
well-scoped change by someone whose context is still warm — graduating it
immediately reverses tech-debt accrual before it cools and keeps the
exclusion list from compounding. Old legacy entries don't get worse by
waiting another week; new ones do.

**How to apply:** this LIFO rule supersedes the older "prefer the smallest
by line count" heuristic in the entry-point stubs. An explicit target named
by the user (or by the `#25` reviewer) always wins over the default
ordering — LIFO only governs the no-argument case.

## Workflow

For each file listed in any `exclude:` block in `.pre-commit-config.yaml` (e.g. `pyright`, `interrogate`, `shellcheck`, `codespell`) **or** under `[tool.pydoclint].exclude` or `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` (pydoclint path excludes and ruff per-file rule ignores are configured there, not in the pre-commit config; note that `per-file-ignores` still runs ruff on the file but suppresses specific rules — it is not a path exclude):

1. **Create a branch**: `chore/lint-cleanup/<module-name>` (e.g., `chore/lint-cleanup/surge-datamodule`)
2. **Run hooks on the file**, e.g. `interrogate`
3. **Auto-fix what you can**: `ruff check --fix` and `docformatter --in-place` handle most formatting issues automatically
4. **Manually fix remaining violations**:
   - `interrogate` missing docstrings: add Sphinx-style docstrings (`:param:`, `:returns:`, `:raises:`) to public functions/classes — matches the `docformatter` config (`style = "sphinx"` in `pyproject.toml`) — and must pass `pydoclint` DOC1xx/DOC2xx/DOC5xx (signature ↔ docstring consistency) **and** ruff `D102`/`D103`/`D107` (must-have-docstring on public methods, functions, and `__init__`).
5. **Remove the file from every exclusion list it appears in.** Check `.pre-commit-config.yaml`'s `exclude:` blocks **and** `pyproject.toml`'s `[tool.pydoclint].exclude` and `[tool.ruff.lint.per-file-ignores]`. A single file may appear in more than one list (e.g. excluded by `interrogate` in pre-commit *and* by `ANN001` per-file-ignore in ruff) — graduating the file means clearing every entry.
6. **Verify**: `pre-commit run --files <file>` passes all hooks
7. **Run tests**: `make test-fast` — the quick CPU suite (excludes `slow`, `gpu`, `mps`, `requires_vst`) must still pass as a smoke check; lint-only changes shouldn't affect behavior
8. **Commit**: Use conventional commits format: `chore(lint): clean up <filename>`
9. **Open PR**: PR body references `#25` with `Refs #25` (not `Fixes`/`Closes` — #25 stays open until every file is done). Check off the file in the issue checklist. Add to "Code Health" project.

## Rules

- One file per PR (or 2-3 closely related files, e.g., a module and its tests)
- Never change logic, signatures, return values, or behavior
- Never add features, refactor algorithms, or rename public APIs
- `# noqa` / `# nosec` only with a justification comment explaining why
- If a file requires functional changes to pass lint (e.g., unused imports that are actually used dynamically), skip it and leave a comment on #25
- Line length is 99 (configured in `pyproject.toml` under `[tool.ruff]`)
- Docstrings follow Sphinx style (`:param:`, `:returns:`, `:raises:`) — matches `docformatter` config (`style = "sphinx"` in `pyproject.toml`) — and must pass `pydoclint` DOC1xx/DOC2xx/DOC5xx (signature ↔ docstring consistency)
- Run `make test-fast` after every file to catch regressions

## Files

See the checkbox list in https://github.com/tinaudio/synth-setter/issues/25

## Done when

- All files removed from exclusion lists
- `pre-commit run -a` passes cleanly
- #25 is closed
