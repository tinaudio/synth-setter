# Lint Cleanup Agent

## Goal

Fix pre-existing lint violations in legacy files one at a time, removing them from the pre-commit exclusion lists in `.pre-commit-config.yaml`. Tracked in #25.

## Scope

Only formatting, docstrings, and lint fixes. **No functional changes.**

## Workflow

For each file listed in the `exclude` blocks of `pyright`, `interrogate`, `shellcheck`, `codespell`, and other hooks in `.pre-commit-config.yaml` (ruff per-file-ignores live in `pyproject.toml`):

1. **Create a branch**: `chore/lint-cleanup/<module-name>` (e.g., `chore/lint-cleanup/surge-datamodule`)
2. **Run hooks on the file**: `e.g. `interrogate\`
3. **Auto-fix what you can**: `ruff` and `docformatter` handle most formatting issues automatically
4. **Manually fix remaining violations**:
   - `interrogate` missing docstrings: add Sphinx-style docstrings (`:param:`, `:returns:`, `:raises:`) to public functions/classes — matches the `docformatter` config (`style = "sphinx"` in `pyproject.toml`)
5. **Remove the file from all `exclude` blocks** in `.pre-commit-config.yaml`
6. **Verify**: `pre-commit run --files <file>` passes all hooks
7. **Run tests**: `make test` — all tests must still pass
8. **Commit**: Use conventional commits format: `chore(lint): clean up <filename>`
9. **Open PR**: Reference #25, check off the file in the issue checklist. Add to "Code Health" project.

## Rules

- One file per PR (or 2-3 closely related files, e.g., a module and its tests)
- Never change logic, signatures, return values, or behavior
- Never add features, refactor algorithms, or rename public APIs
- `# noqa` / `# nosec` only with a justification comment explaining why
- If a file requires functional changes to pass lint (e.g., unused imports that are actually used dynamically), skip it and leave a comment on #25
- Line length is 99 (configured in `pyproject.toml` under `[tool.ruff]`)
- Docstrings follow Sphinx style (`:param:`, `:returns:`, `:raises:`) — matches `docformatter --style=sphinx`
- Run `make test` after every file to catch regressions

## Files

See the checkbox list in https://github.com/ktinubu/synth-permutations/issues/25

## Done when

- All files removed from exclusion lists
- `pre-commit run -a` passes cleanly
- #25 is closed
