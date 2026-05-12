# CLAUDE.md

## Project

synth-setter: Synth inversion, sound matching and preset exploration tools

- Python 3.10+, PyTorch Lightning, Hydra configs
- Data pipeline: distributed shard generation on SkyPilot-managed compute (RunPod + OCI), stored in Cloudflare R2
- Design doc: `docs/design/data-pipeline.md`

## Code Standards

### Formatting & Linting (enforced by pre-commit)

- **Ruff format** (line-length=99)
- **Ruff** (rules: E, F, I, S, T, UP, W, ANN001)
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

#### Comment Hygiene: Don't Bake Values Into Comments

Comments that restate values, counts, or list contents go stale the moment the code changes. The code is the source of truth — name it, don't mirror it.

- **Don't restate constant values.** Next to `num_samples = 6`, never write `# 6 samples (12 renders total)`. Reference the symbol (`each stage renders num_samples times`) or describe behavior without numbers.

- **Don't bake in counts the code already reports.** `# 29 review comments triaged` or `# 5 metric series` belongs in a PR description (a snapshot in time), not in a docstring or inline comment that lives next to the data and will drift the next time someone adds an item. Prefer "the metric series listed below" or reference the data source.

- **Don't enumerate list contents in prose.** Next to `THINGS = ["a", "b", "c"]`, never write `# three things: a, b, and c` — both the count and the contents will mismatch the list within a release. The list is the source of truth; a comment can name the *category*, not its contents.

- **Keep comments terse — typically one short line.** Multi-sentence prose explanations inline are a smell. If a comment would need more than ~2 lines to be useful, that's a signal the context belongs in a GitHub issue, not in the source. Make the inline comment a one-line pointer to the issue.

  ```python
  # Bad — multi-sentence essay inline:
  # We use os._exit(0) here instead of sys.exit(0) because SkyPilot's
  # job runner wraps the process in a shell that doesn't propagate the
  # exit code correctly when atexit handlers raise during interpreter
  # shutdown — see the long investigation in the PR description.
  os._exit(0)

  # Good — one-line pointer to the issue with the full context:
  # Workaround for atexit-during-shutdown hang — see #735.
  os._exit(0)
  ```

- Still write comments for: WHY a non-obvious choice was made, hidden invariants, workarounds (with bug ID), and surprising behavior.

#### No Comments Inside YAML `run:` Block-Scalars

YAML block-scalars passed to bash — i.e. `run: |` blocks in **GitHub Actions workflow YAML** (`.github/workflows/*.{yml,yaml}`) and **SkyPilot Task YAML** (`configs/compute/*.yaml`'s `run:` and `setup:` blocks) — render without syntax highlighting in most YAML viewers and are visually indistinguishable from "real" command lines once they reach bash. Stray `'`, `` ` ``, `$`, or `\` inside a comment have caused unintended shell quoting / expansion in the past.

**Rule:** put comments *above* the block-scalar, not inside it.

```yaml
# Bad — comments INSIDE the run: block:
- name: Pin image tag
  run: |
    # The template's image_id defaults to dev-snapshot; pin it to the
    # tag this run was dispatched with.
    sed -i "s|...dev-snapshot|...${IMAGE_TAG}|" configs/compute/runpod-template.yaml

# Good — comments ABOVE the step:
# Pin the template's image_id from its default (`dev-snapshot`) to the tag
# this run was dispatched with, so the worker pulls the same image we just
# smoke-tested locally in the prior step.
- name: Pin image tag
  run: |
    sed -i "s|...dev-snapshot|...${IMAGE_TAG}|" configs/compute/runpod-template.yaml
```

Same rule for SkyPilot `run:`/`setup:` blocks — put rationale comments at the YAML structural level (above the `run:` key), not inside the block-scalar.

The block-scalar should contain only commands. The reader who wants to know *why* a command exists looks at the comment block above the step or above the `run:` key — outside the bash interpretation surface.

### Testing

- **pytest** with strict markers. Run `make test-fast` for quick tests (CPU-only; excludes slow, gpu, mps, requires_vst).
- Mark slow tests with `@pytest.mark.slow`.
- Write tests for new code. Test behavior, not implementation details.
- Use descriptive test names: `test_<what>_<condition>_<expected>`.

### Architecture

- `src/` — ML code (models, data modules, training, evaluation) and the dataset-generation entrypoint (`src/generate_dataset.py`)
- `src/synth_setter/` — PEP src-layout package; empty scaffold in Phase 1 of [#784](https://github.com/tinaudio/synth-setter/issues/784), receives moved modules across Phases 2–5.
- `src/pipeline/` — distributed data pipeline (`python -m src.pipeline` planned — [#72](https://github.com/tinaudio/synth-setter/issues/72))
  - `schemas/` — Pydantic models (`DatasetSpec` + `RenderConfig` in `spec.py`, `prefix`, `image_config`; planned: report, card, sample — [#74](https://github.com/tinaudio/synth-setter/issues/74))
  - `ci/` — CI validation scripts (materialize_spec, validate_shard, validate_spec, load_image_config)
  - `constants.py` — shared constants (R2 bucket, spec filename)
  - `skypilot_launch.py` — SkyPilot launcher CLI for the distributed pipeline
  - `stages/` — generate and finalize stage logic (planned — [#72](https://github.com/tinaudio/synth-setter/issues/72))
  - `backends/` — compute providers: local, RunPod (planned — [#71](https://github.com/tinaudio/synth-setter/issues/71))
- `scripts/` — standalone scripts
- `configs/` — Hydra YAML configs. `dataset.yaml` is the top-level datagen entrypoint config (mirrors `train.yaml` / `eval.yaml`); see `configs/dataset.yaml`'s `defaults:` for its composition groups
- `tests/` — mirrors `src/` (including `tests/pipeline/` for `src/pipeline/`)
- `docs/design/` — design documents

### Git Workflow

- **Always use isolated git worktrees** for feature work, bug fixes, and PRs. Never edit files directly on a development branch in the main working tree — branch switching and stash conflicts cause lost work and accidental commits to wrong branches.
- Use `isolation: "worktree"` when spawning subagents that write code or create commits.
- The main working tree should only be used for read-only operations (exploration, `git log`, `rclone ls`, etc.).
- When using Claude Code's Agent tool with `isolation: "worktree"`, the worktree is automatically cleaned up if the agent makes no changes. If changes are made, the worktree path and branch are returned for review. For manually created worktrees, clean up with `git worktree remove` when done.
- After `git add -f`, always run `make format` before committing.
- Always verify the correct git branch before pushing commits. Run `git branch --show-current` and confirm it matches the target PR branch before any push.
- **Epic traceability:** Every issue must trace to an Epic via the sub-issue hierarchy (Epic → Phase → Task/Bug/Feature). There are no standalone tasks — all work items need a home. PRs that reference orphan issues lose epic traceability.
- **PR metadata hooks:** The `github-taxonomy` skill enforces taxonomy compliance (type, label, milestone, epic lineage) before every `gh pr create`. The CI workflow `pr-metadata-gate.yaml` provides a second check.

### Pipeline-Specific Rules

- R2 is the source of truth for pipeline state — not metadata or reports.
- Workers only write under `metadata/workers/`. Finalize only writes to `data/`.
- Shard validation is tiered: workers do full 4-check, finalize does structural.
- Never write to `data/shards/` except in finalize.
- Shard IDs are logical (`shard-000042`), deterministic, infrastructure-independent.

## Code Review

Two project-local skills package the review workflow:

- **`/repo-review`** (MVP, default) — single agent, inline core checklist sourced from this CLAUDE.md's hard rules. See `.claude/skills/repo-review/SKILL.md` for the authoritative checklist. No plugin dependency — works on a fresh clone, in CI, for external contributors.
- **`/repo-review-full`** (heavyweight) — fans out one parallel agent per applicable plugin checklist. Selection rules live in `.claude/skills/repo-review-full/SKILL.md`. Requires the `tinaudio-synth-setter-skills` plugin.

Both skills aggregate BLOCK/WARN findings, prefix each with `[<skill>:<severity>]`, and post every one as an individual unresolved inline review comment via `.claude/skills/_shared/post_review.py`. The helper anchors each finding to a line in the diff's hunks; findings whose natural line is outside the hunks fall back to the nearest in-hunk line on the same file with a cross-ref note in the body, and findings on files entirely outside the diff are rolled into the top-level review body.

Canonical checklist reference (full content lives in the plugin):

1. `tdd-implementation` — TDD compliance (16 items)
2. `code-health` — code quality (24 items)
3. `ml-data-pipeline` — ML pipeline (12 items)
4. `synth-setter-project-standards` — project-specific (30 items)
5. `python-style` — Google Python Style Guide (21 items)
6. `shell-style` — Google Shell Style Guide (19 items, `.sh` files + bash inside YAML `run:` blocks)
7. `ml-test` — ML testing (25 items, model/pipeline test code)

Review all changed code against every applicable checklist. Skip style issues (Ruff handles formatting and linting).

## Refactoring

When refactoring or moving code, always grep ALL file types (not just .py) for references to the old path/name before considering the task complete. Include .yaml/.yml, .md, .json, .toml, .sh, and Dockerfile.

## Design Principles

Before implementing a new abstraction or design pattern, confirm the scope and abstraction level with the user. Prefer YAGNI — start minimal and expand only when asked. Do not over-engineer models or specs.

## Implementation Approach

- Always prefer the simplest viable implementation first. No extra abstractions, no speculative generality unless explicitly asked for or specified by design doc.
- Present a plan before writing code. Wait for approval.
- If you're tempted to introduce a new class, config schema, or architectural pattern, ask: "Do we need this now, or is this speculative?" Default to no.
- Refactoring comes later, driven by real needs, not anticipated ones.

### Mandatory Skills for Code Changes

Whenever Claude implements or modifies non-documentation code (anything other than pure `.md` or `docs/` edits), Claude MUST invoke, in order:

1. `/tdd-implementation` — drive the change test-first.
2. `/code-health` — review and clean up the resulting code.
3. `/simplify` — final reuse and efficiency pass.

`/tdd-implementation` and `/code-health` ship in the `tinaudio-synth-setter-skills` plugin (same source as `/repo-review-full`); `/simplify` is a Claude Code built-in. None are defined locally under `.claude/skills/`. If the plugin or built-in is not available in the current environment, note the gap in your response rather than silently skipping the step.

Pure documentation edits (`.md` files, `docs/`) are exempt. There are no other exemptions — this is a hard rule, not a suggestion.

## Workflow Rules

### Commits & Hooks

- Never add `Co-Authored-By` trailers to commit messages.
- Never use `--no-verify` when committing — hooks work in worktrees and must not be skipped.
- After force-pushing (squash, amend, rewrite) to a PR branch, update the PR title and description with `gh pr edit` to match.

### PR & Issue References

- Every PR body must link a taxonomy-compliant issue via `Closes #N`, `Fixes #N`, `Refs #N`, or `Part of #N`.
- Use `Refs #N` (not `Fixes #N` or `Closes #N`) when a PR is a workaround or partial fix — `Fixes` auto-closes the issue.
- In chat responses, use full markdown hyperlinks for PR/issue references: `[#N](https://github.com/tinaudio/synth-setter/issues/N)`. In PR/issue bodies, use bare `Fixes #N` / `Closes #N` / `Refs #N` so GitHub auto-close works.
- Never add "Generated with Claude Code" or similar attribution footers to PRs, commits, issues, or comments.

### PR Titles

A PR title must stand on its own. A reader who is familiar with the project but has not opened the linked issue should be able to tell from the title alone *what part of the system the PR touches and what concrete change it makes*. Reviewers, release-notes consumers, and people scanning `git log` rarely click through to the issue — the title is the only context many of them get.

- **Name the specific subject, not just the action.** If the PR migrates *one specific schema or component*, say which one. "Complete migration" or "fix bug" without a noun is not enough.
- **Don't rely on the issue, PR body, or commit list to disambiguate.** If the title would be ambiguous without that context, it is ambiguous, period.
- **Keep the conventional commit prefix and scope.** The added context goes in the human-readable subject after the colon, not in the scope. The scope is a stable component identifier (`pipeline`, `claude-md`); the specific subject lives in the words after the colon.
- **Stay under the gitlint title limit.** If the natural phrasing won't fit, shorten the action verb ("complete" → "finish", "remove" → "drop") or tighten the subject's phrasing — but never drop the specific subject itself to save characters.

Example:

- Bad: `feat(pipeline)!: complete Hydra migration; remove load_dataset_spec_yaml`
- Good: `feat(pipeline)!: complete dataset_spec Hydra migration; remove load_dataset_spec_yaml`

The bad version forces the reader to ask "Hydra migration of *what*?" — the project has had several. The good version answers that question in the title.

### PR Readiness

A PR is **not ready** — for review, merge, or hand-off — until **all** of these hold:

- **All CI checks pass.** Both required and optional checks must pass — a failing, errored, or still-pending check means not ready.
- **No merge conflicts with the base branch.** `gh pr view <N> --json mergeable -q .mergeable` must report `MERGEABLE`. Anything else means not ready, including `UNKNOWN` (GitHub is still computing mergeability) — keep polling until it resolves to `MERGEABLE` or `CONFLICTING`.
- **Every open review comment has an inline reply.** Every unresolved review thread — human reviewers AND Copilot's automated comments — has either a code change linked by commit SHA or an inline reply with justification. See `### PR Review Comments` below for the reply mechanics.
- **Copilot has generated no new comments since the last push.** Copilot re-reviews after every push, usually finishing within ~60s. The PR is not ready until you have verified Copilot is done and either has zero new comments or every new comment has been addressed.

"I pushed the fix" is not the same as "the PR is ready." After pushing, iterate until all conditions are satisfied — use `/loop` (e.g. `/loop 2m gh pr checks <N>`) or repeated polling, do not stop at the first push:

1. Push the change.

2. Wait for CI to finish: `gh pr checks <N> --watch` or `/loop` the checks command.

3. If any check fails, diagnose, fix, push again, return to step 2.

4. Check mergeability with `gh pr view <N> --json mergeable -q .mergeable`. If `CONFLICTING`, rebase or merge the base branch, resolve the conflict, push, return to step 2. If `UNKNOWN`, GitHub hasn't finished computing — wait and poll again. Only `MERGEABLE` clears this step.

5. Reply inline to every open review comment — list them with `gh api repos/<OWNER>/<REPO>/pulls/<N>/comments --paginate`. If a reply required a code change, push and return to step 2. Use `/pr-review-resolver` to drive this systematically.

6. Wait for Copilot to complete its post-push review (~60s, but allow up to 15 minutes). Two endpoints matter here: inline review comments live at `/pulls/<N>/comments`, and top-level review summaries (including a "no findings" note) live at `/pulls/<N>/reviews`. Check both:

   ```bash
   # Inline review comments (per-line nits)
   gh api repos/<OWNER>/<REPO>/pulls/<N>/comments --paginate \
     --jq '[.[] | select(.user.login | test("[Cc]opilot")) | {id, path, line, body}]'
   # Top-level reviews (overall summary, including a no-findings note)
   gh api repos/<OWNER>/<REPO>/pulls/<N>/reviews --paginate \
     --jq '[.[] | select(.user.login | test("[Cc]opilot")) | {id, state, submitted_at, body}]'
   ```

   If Copilot left new unaddressed inline comments, **or** a new top-level review with actionable content (`state=COMMENTED`/`CHANGES_REQUESTED` with a non-empty body that isn't just a "no findings" note), return to step 5 and address it the same way you would inline comments. If 15 minutes have elapsed since the push and Copilot has produced neither an inline comment nor a top-level review note explicitly stating it has no findings, treat the auto-review as not triggered and manually re-request it before continuing — see step 6a below.

   **Step 6a — Manually re-request a Copilot review** when step 6's 15-minute window elapses with no Copilot activity. Try in this order, stopping at the first one that succeeds:

   1. Re-request via the reviewers API (equivalent of clicking the re-request button):
      ```bash
      gh api --method POST \
        /repos/<OWNER>/<REPO>/pulls/<N>/requested_reviewers \
        -f 'reviewers[]=copilot-pull-request-reviewer[bot]'
      ```
      If that errors, confirm the exact bot slug your org uses with `gh pr view <N> --json reviewRequests,reviews` and retry with the correct login.
   2. If the reviewers API still won't take Copilot as a reviewer, force a re-trigger with an empty commit (works when Copilot is wired to run on push rather than as a requested reviewer):
      ```bash
      git commit --allow-empty -m "chore: trigger copilot review"
      git push
      ```
      Pushing restarts the readiness loop — return to step 2.

   After re-requesting, wait another ~60s (allow up to 15 minutes again) and re-check Copilot's comments. Repeat at most once; if Copilot still produces nothing after a manual re-request, record that in the PR thread and move on.

7. Only when checks are all green AND `mergeable=MERGEABLE` AND every review comment has an inline reply AND Copilot has produced no new comments since the last push (or has been confirmed silent via step 6a) is the PR ready.

This applies whether the PR is yours or one you were asked to drive across the finish line.

### PR Verification

- `/pr-checkbox` is verification-only — look for existing branches/PRs and run checks, never plan implementation.
- Each verification step gets a checkbox (`- [ ]` / `- [x]`) with the command run and its console output as evidence.
- Size output appropriately: small (\<20 lines) inline, medium (20-100) in a PR comment, large (100+) in a Gist linked from a comment.
- Only tick `[x]` if the result unambiguously passes.

### GPU Verification

Before skipping a GPU-gated check with "no GPU available" or similar, run BOTH probes and paste their output into the SKIP rationale:

```bash
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

Only skip as "no GPU available" if both probes indicate no usable CUDA GPU: `nvidia-smi` exits non-zero and `torch.cuda.is_available()` returns `False`. If the probes disagree, do not call it "no GPU available" — document it as an environment/setup mismatch (for example, driver/tooling visibility vs. PyTorch CUDA availability) and include both outputs in the rationale. "I assumed there's no GPU" is not justification — the assumption has been wrong before.

### PR Review Comments

- Always reply to PR review comments after pushing a fix — never push silently.
- Reply **inline on the specific review comment** that the change addresses (using `gh api repos/{owner}/{repo}/pulls/comments/{comment_id}/replies` or the equivalent threaded reply), not as a generic top-level PR comment. One inline reply per comment addressed.
- Link the specific fix commit SHA in the reply (e.g., "Fixed in abc1234").
- If a pushed change addresses multiple review comments, post a separate inline reply on each — do not consolidate them into a single comment elsewhere on the PR.

### GitHub Project

- Select the GitHub project by name (`"synth-setter"`) or known ID (`PVT_kwDOD6Bkms4BSS3h`) — never use `nodes[0]` or array index.

## Don't

- Don't modify `.env` (contains real credentials).
- Don't commit `.env`, credentials, or API keys.
- Don't commit without explicit permission.
- Don't run `make docker-*` or RunPod commands without asking first.
- Don't add unnecessary abstractions — only abstract when there are two concrete uses.
- Don't add comments to code you didn't change.

## Commands

```bash
make test-fast         # Quick CPU-only tests (excludes slow, gpu, mps, requires_vst)
make test-full-cpu     # All CPU tests (slow + requires_vst included; gpu/mps excluded)
make test-full-gpu     # GPU + CPU tests (mps excluded). Serial.
make test-full-mps     # MPS + CPU tests (gpu excluded). Serial.
make test-vst-cpu      # VST-only suite (requires_vst, slow included; gpu/mps excluded)
make format            # Run all pre-commit hooks
make clean             # Clean autogenerated files
make help              # Show all targets
```
