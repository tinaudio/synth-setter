# Lint Cleanup Agent

## Goal

Fix pre-existing lint violations in legacy files one at a time, draining
`.pydoclint-baseline.txt` and the surviving entries in `.pre-commit-config.yaml`
exclusion blocks and `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]`.

- pydoclint cleanup is tracked in
  [#938](https://github.com/tinaudio/synth-setter/issues/938).
- Broader lint-cleanup (pyright, interrogate, shellcheck, codespell, ruff
  per-file-ignores) is tracked in
  [#25](https://github.com/tinaudio/synth-setter/issues/25).

[#1044](https://github.com/tinaudio/synth-setter/pull/1044) swapped the
pydoclint cleanup target from `[tool.pydoclint].exclude` regex entries to
rows in `.pydoclint-baseline.txt`. The pydoclint exclude list is now
**infra-only** (matches `.git/`, `.venv/`, `build/`, `dist/`,
`node_modules/`, `.claude/`, `.worktrees/`, `notebooks/`,
`tests/fixtures/`) and **must not be edited** by this workflow. Legacy
docstring violations live as rows in `.pydoclint-baseline.txt`; cleanup
removes rows.

## How this is invoked

This runbook is the canonical workflow. Three entry points delegate to it; edits to the steps below land here and reach every entry point automatically.

| Tool                       | Entry point                        | How to invoke                                                                         |
| -------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| Claude Code (programmatic) | `.claude/agents/lint-cleanup.md`   | `Agent(subagent_type: "lint-cleanup", isolation: "worktree", prompt: "...")`          |
| Claude Code (interactive)  | `.claude/commands/lint-cleanup.md` | Type `/lint-cleanup <path>` in a Claude Code session                                  |
| Copilot Coding Agent       | `.github/copilot-instructions.md`  | Assign a #938 (pydoclint) or #25 (broader lint) sub-issue to Copilot in the GitHub UI |

The entry-point stubs may surface a short cross-reference of the rules most often forgotten (commit prefix, `Refs #938` / `Refs #25`, isolated-worktree requirement) so a contributor seeing only the stub still gets the load-bearing constraints. They must not paraphrase or fork the workflow steps themselves — those live here, single-sourced.

## Scope

Only formatting, docstrings, and lint fixes. **No functional changes.**

## Picking the next file

When the agent has discretion over which file to work next — i.e. invoked
without a specific target path — **pick a file from `.pydoclint-baseline.txt`,
preferring files with the smallest violation count first.**

Group the baseline file's rows by the path header (the bare path line above
each `DOC1xx`/`DOC2xx`/`DOC5xx`/`DOC6xx` block, separated by
`--------------------`) and rank candidates by row count ascending. Skip
files already in flight (open `chore/lint-cleanup/*` PR or a ticked box on
[#938](https://github.com/tinaudio/synth-setter/issues/938)). If two files
tie on row count, break the tie with the smaller file by line count.

The LIFO ("most recently added exclude entry first") heuristic that
governed the pre-#1044 workflow no longer applies to pydoclint. The
pydoclint exclude list is infra-only after #1044 and doesn't accrue legacy
entries anymore, so there is no recency signal there. Smallest-by-row-count
biases toward the most reviewable PRs.

The LIFO rule **still applies** to the other lint stacks tracked under
[#25](https://github.com/tinaudio/synth-setter/issues/25) — entries in
`.pre-commit-config.yaml` `exclude:` blocks (pyright, interrogate,
shellcheck, codespell) and `[tool.ruff.lint.per-file-ignores]` for
non-pydoclint rules. For those, rank by `git blame` timestamp against the
exclusion-list lines and pick the most-recently-added entry first; this
heuristic is unchanged from the pre-#1044 runbook because those exclude
lists *do* still accrue legacy entries.

**`git blame` is a best-effort heuristic, not a perfect provenance signal.**
It reports the *last commit that touched the line*, not necessarily the
commit that originally introduced the entry — a reformat, a YAML reorder,
or an unrelated `exclude:` edit nearby can move the blame timestamp without
the entry actually being "new." That's fine: the goal here is to bias
toward recently-touched entries (which are usually genuinely recent and
context-fresh), not to reconstruct strict introduction order. Do not reach
for `git log -L` or pickaxe in the default path — the extra fidelity isn't
worth the runtime cost or the runbook complexity. If a particular entry
looks misranked, just pick the next one down and move on.

**Why smallest-first for pydoclint baseline rows:** the baseline is a flat
list of `(file, function, rule)` rows; recency information was lost when
the rows were generated together by `pydoclint --generate-baseline=1` in
#1044. The remaining axis worth biasing on is reviewability — small
files, small PRs, fewer chances for the docstring rewrite to incidentally
touch unrelated code.

**Explicit target wins.** A target named by the user (or by the
#938/#25 reviewer) always supersedes the default ordering — the
smallest-first / LIFO rules only govern the no-argument case.

**The `[tool.pydoclint].exclude` regex is off-limits.** Do not remove any
of its bare-entry lines (`^\.git/`, `^build/`, `^notebooks/`,
`^tests/fixtures/`, etc.) as part of this workflow — they are infra paths,
not legacy violations to clean up. If you believe an exclude entry should
be graduated, raise it on
[#938](https://github.com/tinaudio/synth-setter/issues/938) for a
maintainer call rather than touching it inline.

## Workflow

### For pydoclint baseline rows (the common case under #938)

01. **Create a branch**: `chore/lint-cleanup/<module-name>` (e.g.,
    `chore/lint-cleanup/surge-datamodule`).

02. **Inspect the file's baseline rows**: open `.pydoclint-baseline.txt`
    and read every row under the file's path header. Each row names a
    function/method and a rule code
    (`DOC101`/`DOC103`/`DOC201`/`DOC203`/`DOC501`/`DOC503`/`DOC601`/`DOC603`)
    — that's the violation you need to fix.

03. **Run pydoclint against the file directly to see live violations**:
    `pydoclint <path>`. Compare its output to the rows in the baseline —
    they should match. If pydoclint reports *fewer* violations on the file
    than the baseline lists, the baseline is stale for this file
    (asymmetric-shrink case — see step 6).

04. **Auto-fix what you can**: `ruff check --fix <path>` and
    `docformatter --in-place <path>` handle most formatting issues
    automatically.

05. **Manually fix remaining violations**:

    - `DOC1xx` (args): add or repair `:param <name>:` entries so the
      docstring matches the signature exactly. Sphinx style — matches
      `[tool.docformatter] style = "sphinx"` in `pyproject.toml`.
    - `DOC2xx` (returns): add `:returns:` (and the type) when the function
      has a non-`None` return annotation.
    - `DOC5xx` (raises): add `:raises <ExceptionType>:` for each exception
      raised in the body (`skip-checking-raises = false`).
    - `DOC6xx` (class attrs): document each class attribute via a class
      docstring attribute table or matching `:ivar:` entries.
    - Ruff D-family rules are also enforced. After #1044, the set is
      `D100`/`D101`/`D102`/`D103`/`D107`/`D205`/`D401` — `D100` (module
      docstring), `D101` (class docstring), `D205` (blank line between
      summary and description), and `D401` (imperative-mood summary) bite
      in the rewrite step. Read the **D205 sentence-bisection warning**
      below before mechanically inserting blank lines.

06. **Regenerate the baseline so the rows you fixed are dropped**:

    ```bash
    pydoclint --generate-baseline=1 src/ tests/ scripts/
    ```

    Then inspect `git diff .pydoclint-baseline.txt`. The diff **must show
    only deletions** for the file you worked on, never additions.

    - If the diff shows additions (anywhere, not just on your file), the
      change introduced a new violation. Stop, do not commit — diagnose
      the new violation, fix it, regenerate, and re-check.
    - If the diff shows deletions on files you did not touch, the baseline
      was already out of sync with `main` (the asymmetric-shrink case).
      **Do not silently absorb those deletions into your PR.** Reference
      [#1055](https://github.com/tinaudio/synth-setter/issues/1055) — the
      ticket tracking the weekly regenerate-and-PR check for stale
      baseline rows — in a comment on the PR, and restrict your diff to
      deletions on the file you intended to fix (e.g., by hand-editing
      `.pydoclint-baseline.txt` back to drop only the rows you actually
      addressed). #1055 owns the global stale-row sweep; do not duplicate
      its work inside a cleanup chunk.

07. **Verify**: `pre-commit run --files <path>` passes all hooks. The
    pydoclint hook reads `.pydoclint-baseline.txt`; `auto-regenerate-baseline = false`
    means the hook will fail if your fix introduced a new violation, but
    it will not silently re-shrink the baseline when you fix one.

08. **Run tests**: `make test-fast` — the quick CPU suite (excludes
    `slow`, `gpu`, `mps`, `requires_vst`) must still pass as a smoke
    check; lint-only changes shouldn't affect behavior.

09. **Commit**: Use conventional commits format:
    `chore(lint): clean up <module-name> baseline rows` (or
    `chore(lint): drain <module-name> from pydoclint baseline`). The diff
    must include both the docstring fixes **and** the
    `.pydoclint-baseline.txt` row deletions in a single commit.

10. **Open PR**: PR body references `#938` with `Refs #938` (not
    `Fixes`/`Closes` — #938 stays open until the baseline reaches 0
    rows). Include a one-line summary near the top:
    `shrinks baseline by N rows` (where N is the deletion count from
    step 6's diff). Tick the file in the issue checklist. Add to "Code
    Health" project.

### For other exclusion lists (pyright, interrogate, shellcheck, codespell, ruff per-file-ignores under #25)

These lists are *not* covered by the pydoclint baseline mechanism. The
pre-#1044 workflow still applies:

1. Pick a file (LIFO via `git blame` — see **Picking the next file**).
2. Branch `chore/lint-cleanup/<module-name>`.
3. Run the hook on the file, auto-fix what you can, hand-fix the rest.
4. **Remove the file from every exclusion list it appears in.** A single
   file may appear in more than one list (e.g. excluded by `interrogate`
   in pre-commit *and* by `ANN001` per-file-ignore in ruff) — graduating
   the file means clearing every entry. **Do not touch `[tool.pydoclint].exclude`**
   — that list is infra-only after #1044. Pydoclint cleanup happens via
   the baseline-row path above, not via exclude removal.
5. `pre-commit run --files <path>` green; `make test-fast` green.
6. Commit `chore(lint): clean up <filename>`; PR body `Refs #25`.

## D205 sentence-bisection warning

`D205` requires a blank line between the docstring summary and any
description block. A mechanical fix — "find a too-long single-line
docstring, insert a blank line in the middle of the sentence" — produces
an ungrammatical sentence fragment as the summary. This is the dominant
maintainability regression seen in the #1044 review (Copilot comment
[3243241015](https://github.com/tinaudio/synth-setter/pull/1044#discussion_r3243241015)
on `src/synth_setter/tools/surge_xt_interactive.py` flagged the canonical
bad pattern).

**Do not split a sentence in the middle to satisfy `D205`. Rewrite the
first line as a complete imperative sentence under ~95 chars and demote
the rest to a body paragraph after the blank line.**

Bad — mechanical bisection (Copilot's flagged pattern):

```python
def parse(s: str) -> PredictionRef:
    """Parse a ``PATH:BATCH_IDX`` string into a.

    ``PredictionRef``.
    """
```

The summary ``` "Parse a ``PATH:BATCH_IDX`` string into a." ``` ends with a
stray period after `"a"` and reads as a sentence fragment. The reader
has to glue the two halves back together to recover the meaning.

Good — full imperative summary, real description block only when needed:

```python
def parse(s: str) -> PredictionRef:
    """Parse a ``PATH:BATCH_IDX`` string into a ``PredictionRef``."""
```

Or, if elaboration is genuinely warranted:

```python
def parse(s: str) -> PredictionRef:
    """Parse a ``PATH:BATCH_IDX`` string into a ``PredictionRef``.

    The path may be absolute or relative; the batch index is an integer
    offset into the prediction tensor.
    """
```

If the natural imperative summary won't fit under ~95 chars, **shorten
the verb or tighten the noun phrase** — do not bisect. "Compute the …"
can become "Return …"; "Helper for parsing …" can become "Parse …". The
summary must stand on its own as a sentence.

The same rule applies to `D401`: rewriting `"Returns the foo."` to
`"Return the foo."` is fine; rewriting it by splitting the sentence
across lines to "satisfy" the rule is not.

## Rules

- One file per PR (or 2–3 closely related files, e.g., a module and its
  tests). For pydoclint cleanups, the PR includes both the docstring
  fixes and the corresponding `.pydoclint-baseline.txt` row deletions in
  a single commit.
- Never change logic, signatures, return values, or behavior.
- Never add features, refactor algorithms, or rename public APIs.
- Never edit the `[tool.pydoclint].exclude` regex in `pyproject.toml` —
  it is infra-only after #1044. Pydoclint cleanup happens via baseline-row
  deletions, not exclude-list edits.
- `# noqa` / `# nosec` only with a justification comment explaining why.
  For `D401` on placeholder command stubs that genuinely name a noun (not
  an action), `# noqa: D401` is acceptable — there are two existing
  examples in `src/synth_setter/tools/docker_entrypoint.py`.
- If a file requires functional changes to pass lint (e.g., unused
  imports that are actually used dynamically), skip it and leave a
  comment on [#938](https://github.com/tinaudio/synth-setter/issues/938)
  (or [#25](https://github.com/tinaudio/synth-setter/issues/25) for
  non-pydoclint stacks).
- Line length is 99 (configured in `pyproject.toml` under `[tool.ruff]`).
- Docstrings follow Sphinx style (`:param:`, `:returns:`, `:raises:`) —
  matches `docformatter` config (`style = "sphinx"` in `pyproject.toml`)
  — and must pass `pydoclint` `DOC1xx`/`DOC2xx`/`DOC5xx` (signature ↔
  docstring consistency) **and** ruff `D100`/`D101`/`D102`/`D103`/`D107`
  (must-have-docstring on modules/classes/methods/functions/`__init__`)
  **and** `D205`/`D401` (summary shape and imperative mood — see the
  bisection warning above).
- Run `make test-fast` after every file to catch regressions.

## Files

- pydoclint baseline rows: the authoritative source-of-truth for "what's
  left" is `.pydoclint-baseline.txt` itself. Group its rows by path
  header to see which files still have violations and how many. See also
  the checkbox list in
  [#938](https://github.com/tinaudio/synth-setter/issues/938).
- Other exclusion lists: see the checkbox list in
  [#25](https://github.com/tinaudio/synth-setter/issues/25).

## Done when

- `.pydoclint-baseline.txt` is empty (or contains only rows for files
  that have been judged out-of-scope on #938).
- All files removed from non-pydoclint exclusion lists tracked in #25.
- `pre-commit run -a` passes cleanly.
- Both [#938](https://github.com/tinaudio/synth-setter/issues/938) and
  [#25](https://github.com/tinaudio/synth-setter/issues/25) are closed.
