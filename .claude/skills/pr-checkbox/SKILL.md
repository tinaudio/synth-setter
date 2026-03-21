---
name: pr-checkbox
description: Rigid process skill for posting PR verification results. Use this skill whenever running verification steps on a PR, posting validation results to a PR, or claiming that checks pass. Also trigger when writing verification sections in PR descriptions, commenting verification outcomes, or updating verification checkboxes. If you are about to write "PASS" or "FAIL" or a verification table on a PR, you MUST use this skill first. Trigger on any PR review, validation, or verification activity.
---

# PR Verification Checkbox Skill

This is a **rigid** process skill. Follow every step exactly — no shortcuts, no "PASS" assertions without evidence.

## Why this matters

Verification without evidence is just opinion. When someone reads a PR months later, they need to see exactly what was run and what came back — not a table of "PASS" labels. Checkboxes with commands and output make verification auditable, reproducible, and trustworthy.

## The Rule

Every verification step gets a checkbox. Every checkbox shows the command and its output. The checkbox is only ticked if the result is unambiguous.

## Process

### Step 1: Run each verification step and capture output

For every verification step defined in the PR:

1. Run the exact command
2. Capture the full console output
3. Determine: does this **unambiguously** pass, fail, or is it ambiguous?

### Step 2: Format results with checkboxes

Each verification step becomes a checkbox item. The format depends on output size:

#### Small output (< ~20 lines) — inline in PR description

Put the command and output directly next to the checkbox:

```markdown
- [x] **JSON syntax valid**
  ```bash
  $ python3 -c "import json; json.load(open('pyrightconfig.json'))"
  # (no output — exit code 0)
  ```

- [x] **Threshold is 1% not 100%**
  ```bash
  $ grep threshold .github/codecov.yml
  threshold: 1%  # fail if coverage drops by more than 1%
  threshold: 1%  # fail if new code has >1% less coverage than target
  ```

- [ ] **Type checking passes** *(ambiguous — see comment)*
  ```bash
  $ pyright --project pyrightconfig.json
  0 errors, 3 warnings, 0 informations
  ```
  > Warnings present — needs review to determine if these are expected.
```

#### Medium output (20-100 lines) — PR comment with link

1. Post a PR comment containing the command and full output
2. In the PR description checkbox, link to that comment:

```markdown
- [x] **All tests pass in parallel mode** — [verification output](#issuecomment-12345)
```

The comment itself should contain:

```markdown
## Verification: All tests pass in parallel mode

```bash
$ pytest -n auto -m "not slow" -vv
========================= test session starts ==========================
platform darwin -- Python 3.10.12, pytest-7.4.0, pluggy-1.2.0
...
========================= 42 passed in 12.34s ==========================
```
```

#### Large output (100+ lines) — GitHub Gist with summary

1. Create a GitHub Gist with the full console output
2. Post a PR comment with a summary, key lines, and link to the gist
3. In the PR description checkbox, link to the comment:

```markdown
- [x] **Full test suite passes** — [summary + gist](#issuecomment-67890)
```

The comment should contain:

```markdown
## Verification: Full test suite passes

**Result:** 142 passed, 0 failed, 3 skipped

Key output:
```bash
$ pytest -n auto
...
========================= 142 passed, 3 skipped in 45.12s =============
```

Full output: https://gist.github.com/user/abc123
```

### Step 3: Tick or leave unchecked

- **`[x]` (ticked):** The output unambiguously confirms the check passes. Zero doubt.
- **`[ ]` (unchecked):** Any ambiguity at all. Add a brief explanation of what's unclear:
  - Warnings present that might or might not matter
  - Output doesn't cleanly confirm the expected result
  - Exit code 0 but output contains error-like text
  - Test passed but with unexpected side effects in the output

When in doubt, leave unchecked. A false `[x]` is worse than a cautious `[ ]`.

### Step 4: Post to the PR

Place the verification checklist in the PR description (if writing it for the first time) or as a comment (if verifying after the PR was created). Use a clear header:

```markdown
## Verification Results
```

## Creating Gists for large output

When output exceeds ~100 lines, create a gist:

```bash
gh gist create --public -f "verification-<step-name>.log" /path/to/output.log
```

Include the gist URL in your PR comment alongside the summary.

## Anti-patterns — do NOT do these

- Writing a table with "PASS" / "FAIL" columns but no commands or output
- Ticking `[x]` when output contains warnings you haven't investigated
- Saying "all checks pass" without showing what was run
- Posting verification results without the actual console output
- Summarizing output as "looks good" instead of showing it
