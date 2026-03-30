# CLAUDE.md

## Project

synth-setter: Synth inversion, sound matching and preset exploration tools

- Python 3.10+, PyTorch Lightning, Hydra configs
- Data pipeline: distributed shard generation on RunPod, stored in Cloudflare R2
- Design doc: `docs/design/data-pipeline.md`

## Code Standards

### Formatting & Linting (enforced by pre-commit)

- **Ruff format** (line-length=99)
- **Ruff** (rules: E, F, I, S, T, UP, W)
- Run `make format` before committing

### Commit Messages

Conventional commits, enforced by gitlint (`.gitlint` config). Prefix matters for semantic versioning:

**Version-bumping prefixes:**

- `feat:` → **minor** bump. New user-facing capability (new model, new pipeline stage, new CLI command). The feature must be usable after this commit.
- `fix:` / `perf:` / `revert:` → **patch** bump. Bug fixes, performance improvements, and reverts (reverts are roll-forwards on an append-only main).
- `feat!:` or `BREAKING CHANGE:` footer → **major** bump. Coordinate with a maintainer.

**No-bump prefixes:**

- `internal-feat:` → new code building toward a feature not yet exposed to users (new internal API, new module, new config schema). Use this when a feature is being built across multiple PRs and this PR adds real, tested code — but the feature isn't user-facing yet. No version bump.
- `internal-fix:` → fix to internal code not yet exposed to users. No version bump.
- `monitoring:`, `docs:`, `chore:`, `ci:`, `test:`, `refactor:`, `style:`, `build:` → no version bump.

**When to use which:**

- Each PR should leave main in a valid state — no dead code, no unhooked partial implementations.
- If the PR adds new tested code that will be consumed later, use `internal-feat:`.
- The PR that wires everything together and makes the feature user-facing uses `feat:`.
- Don't contort prefixes to avoid bumps. If it's user-facing, it's `feat:`.

### Writing Code

- Write readable code. Prefer clarity over cleverness.
- Type-annotate all function signatures. Avoid `Any` — use `Union`, `Optional`, or specific types.
- No bare `except:` — always catch specific exceptions.
- Pydantic `BaseModel` with `strict=True` at trust boundaries (config parsing, JSON from R2, worker reports). Dataclasses for internal typed containers.
- Keep functions short and single-purpose. If a function needs a comment explaining what a block does, extract it.
- Use `structlog` for logging in pipeline code. Use Python's `logging` module elsewhere.
- All `rclone` operations use `--checksum`.

### Testing

- **pytest** with strict markers. Run `make test` for quick tests (excludes slow).
- Mark slow tests with `@pytest.mark.slow`.
- Write tests for new code. Test behavior, not implementation details.
- Use descriptive test names: `test_<what>_<condition>_<expected>`.

### Architecture

- `src/` — ML code (models, data modules, training, evaluation)
- `pipeline/` — distributed data pipeline (`python -m pipeline`)
  - `schemas/` — Pydantic models (config, spec, report, card, sample)
  - `stages/` — generate and finalize stage logic
  - `backends/` — compute providers (local, RunPod)
- `scripts/` — standalone scripts
- `configs/` — Hydra YAML configs (`data/`, `trainer/`) and pipeline configs (`dataset/`)
- `tests/` — mirrors `src/` and `pipeline/` structure
- `docs/design/` — design documents

### Git Workflow

- **Always use isolated git worktrees** for feature work, bug fixes, and PRs. Never edit files directly on a development branch in the main working tree — branch switching and stash conflicts cause lost work and accidental commits to wrong branches.
- Use `isolation: "worktree"` when spawning subagents that write code or create commits.
- The main working tree should only be used for read-only operations (exploration, `git log`, `rclone ls`, etc.).
- When using Claude Code's Agent tool with `isolation: "worktree"`, the worktree is automatically cleaned up if the agent makes no changes. If changes are made, the worktree path and branch are returned for review. For manually created worktrees, clean up with `git worktree remove` when done.
- **Submodules:** Skills live in a git submodule at `.claude/skills/` (from `tinaudio/skills`). Clone with `--recurse-submodules`. In new worktrees, run `git submodule update --init`.
- Always verify the correct git branch before pushing commits. Run `git branch --show-current` and confirm it matches the target PR branch before any push.
- **Epic traceability:** Issues that participate in the roadmap hierarchy must be created as sub-issues of the appropriate Phase or Epic. Standalone tasks explicitly allowed by `docs/design/github-taxonomy.md` are exempt from this requirement but must follow that document's guidance (labels, milestones, etc.). PRs that reference orphan roadmap issues lose epic traceability.
- **PR metadata hooks:** The `github-taxonomy` skill enforces taxonomy compliance (type, label, milestone, epic lineage), respecting the standalone-task exceptions defined in `docs/design/github-taxonomy.md`, before every `gh pr create`. The CI workflow `pr-metadata-gate.yaml` provides a second check.

### Pipeline-Specific Rules

- R2 is the source of truth for pipeline state — not metadata or reports.
- Workers only write under `metadata/workers/`. Finalize only writes to `data/`.
- Shard validation is tiered: workers do full 4-check, finalize does structural.
- Never write to `data/shards/` except in finalize.
- Shard IDs are logical (`shard-000042`), deterministic, infrastructure-independent.

## Code Review

When reviewing code or PRs, invoke these skills in order:

1. `tdd-implementation` — TDD compliance checklist (16 items)
2. `code-health` — code quality checklist (24 items)
3. `ml-data-pipeline` — ML pipeline checklist (12 items)
4. `project-standards` — project-specific checklist (30 items)
5. `python-style` — Google Python Style Guide checklist (21 items)
6. `shell-style` — Google Shell Style Guide checklist (19 items, `.sh` files only)
7. `ml-test` — ML testing checklist (25 items, model/pipeline test code)

Review all changed code against every checklist. Prefix findings with BLOCK: (must fix) or WARN: (advisory). Skip style issues (Ruff handles formatting and linting).

## Refactoring

When refactoring or moving code, always grep ALL file types (not just .py) for references to the old path/name before considering the task complete. Include .yaml/.yml, .md, .json, .toml, .sh, and Dockerfile.

## Design Principles

Before implementing a new abstraction or design pattern, confirm the scope and abstraction level with the user. Prefer YAGNI — start minimal and expand only when asked. Do not over-engineer models or specs.

## Implementation Approach

- Always prefer the simplest viable implementation first. No extra abstractions, no speculative generality unless explicitly asked for or specified by design doc.
- Present a plan before writing code. Wait for approval.
- If you're tempted to introduce a new class, config schema, or architectural pattern, ask: "Do we need this now, or is this speculative?" Default to no.
- Refactoring comes later, driven by real needs, not anticipated ones.

## Don't

- Don't modify `.env` (contains real credentials).
- Don't commit `.env`, credentials, or API keys.
- Don't commit without explicit permission.
- Don't run `make docker-*` or RunPod commands without asking first.
- Don't add unnecessary abstractions — only abstract when there are two concrete uses.
- Don't add comments to code you didn't change.

## Commands

```bash
make test              # Quick tests (excludes slow)
make test-full         # All tests
make format            # Run all pre-commit hooks
make clean             # Clean autogenerated files
make help              # Show all targets
```

<!-- plumb:start -->

## Plumb (Spec/Test/Code Sync)

This project uses Plumb to keep the spec, tests, and code in sync.

- **Spec:** Paths configured in `.plumb/config.json` under `spec_paths`
- **Tests:** tests/
- **Decision log:** `.plumb/decisions/`

### Setup

After cloning (or in a new worktree), install `plumb-dev` and run `plumb init`
to install the Plumb git hooks. Note: `plumb init` writes a native git hook to
`.git/hooks/pre-commit` that is separate from the `pre-commit` framework. After
running `plumb init`, re-run `pre-commit install` so both hook systems are
chained. Run `make format` before committing to ensure all linting hooks execute.

### When working in this project:

- Run `plumb status` before beginning work to understand current alignment.
- Run `plumb diff` before committing to preview what Plumb will capture.
- When `git commit` is intercepted by Plumb, **use `AskUserQuestion`** to present
  each pending decision via the native multiple-choice UI. Options: Approve,
  Ignore, Reject. Then run the corresponding `plumb` command.
  **NEVER approve, reject, or edit decisions on the user's behalf.** This is
  non-negotiable.
- After all decisions are resolved, run `plumb sync` to update the spec and
  generate tests. Stage the sync output, then re-run `git commit`. Draft the
  commit message **after** decision review and include a list of approved
  decisions.
- Use `plumb coverage` to identify what needs to be implemented or tested next.
- Never edit files in `.plumb/decisions/` directly.
- Treat the spec markdown files as the source of truth for intended behavior.
  Plumb will keep them updated as decisions are approved.

<!-- plumb:end -->
