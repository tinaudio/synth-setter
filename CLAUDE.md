# CLAUDE.md

## Project

synth-setter: Synth inversion, sound matching and preset exploration tools

- Python 3.10+, PyTorch Lightning, Hydra configs
- Data pipeline: distributed shard generation on RunPod, stored in Cloudflare R2
- Design doc: `docs/design/distributed-pipeline.md`

## Code Standards

### Formatting & Linting (enforced by pre-commit)

- **Black** (line-length=99)
- **Ruff** (rules: E, F, W, I, UP, D417; docstrings)
- Run `make format` before committing

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
- `scripts/` — pipeline orchestration (generate, finalize, RunPod launch)
- `configs/` — Hydra YAML configs (pipeline, data, trainer)
- `tests/` — mirrors `src/` and `scripts/` structure
- `docs/design/` — design documents

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

Review all changed code against every checklist. Prefix findings with BLOCK: (must fix) or WARN: (advisory). Skip style issues (Black/Ruff handle formatting).

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
make test-bash         # BATS shell script tests
make test-pipeline     # Pipeline tests only
make test-models       # Model tests only
make format            # Run all pre-commit hooks
make help              # Show all targets
```
