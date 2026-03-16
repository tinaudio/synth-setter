---
name: review
description: Comprehensive code review orchestrating seven standards checklists. Invoke with /review for local reviews.
---

# Code Review

You MUST complete all four steps below in order.

## Step 1: Load All Checklists

You MUST invoke each of these seven skills using the Skill tool before reviewing any code.
Each skill contains a detailed checklist. Do not skip any. Do not summarize from memory.

1. **Invoke skill: `tdd-implementation`** — TDD Compliance Report checklist (16 items: Red-Green-Refactor, behavior testing, pytest-mock, test naming, mutation testing)
2. **Invoke skill: `code-health`** — Code Health Review Report checklist (24 items: nesting, data flow, cognitive load, booleans, comments, YAGNI, magic numbers, few arguments, Law of Demeter, DRY, dependency injection, code smells, domain objects, conventional commits, commit size, branch naming)
3. **Invoke skill: `ml-data-pipeline`** — ML Pipeline Code Review checklist (12 items: pure transforms, config externalization, type annotations, shape contracts, audio config)
4. **Invoke skill: `project-standards`** — project-specific checklist (30 items: type safety, error handling, pipeline invariants, security, HDF5/numpy, logging)
5. **Invoke skill: `python-style`** — Python style checklist (21 items: imports, exceptions, naming, docstrings, module docstrings, type annotations, TypeVar, mutable defaults, nested functions, default iterators, resources)
6. **Invoke skill: `shell-style`** — Shell style checklist (19 items: quoting, tests, arithmetic, arrays, error handling, `(( ))` / `set -e` caveat, function comments) — apply only to `.sh` files and bash scripts
7. **Invoke skill: `ml-test`** — ML testing checklist (25 items: output shapes, data leakage, loss at init, backprop dependencies, overfit single batch, invariance/directional tests, pipeline test granularity, additive vs retroactive, probabilistic tension, human baseline, Karpathy's recipe) — apply to ML model and pipeline test code

## Step 2: Gather Changes

Run `git diff` and `git diff --cached` to see all current changes.
If a diff was already provided inline (e.g., piped from a commit hook), use that instead.

## Step 3: Review Against All Checklists

Go through EVERY checklist item from ALL seven loaded skills. Evaluate ONLY the changed code.

- Skip style issues — Black and Ruff handle formatting.
- Be strict. If a checklist item is violated, flag it.
- Not every checklist item applies to every diff — skip items that don't apply to the changed code.

## Step 4: Output

For each issue found, output one line:

```
BLOCK: file:line — [category] description
WARN: file:line — [category] description
```

Categories: `tdd`, `code-health`, `ml-pipeline`, `project`, `python-style`, `shell-style`, `ml-test`

End with:

```
Summary: X BLOCK, Y WARN
```

If no issues found, output: `PASS`
