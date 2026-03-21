---
name: pr-checkbox
description: Rigid process skill for posting PR verification results. Use this skill whenever running verification steps on a PR, posting validation results to a PR, or claiming that checks pass. Also trigger when writing verification sections in PR descriptions, commenting verification outcomes, or updating verification checkboxes. If you are about to write "PASS" or "FAIL" or a verification table on a PR, you MUST use this skill first. Trigger on any PR review, validation, or verification activity.
---

# PR Verification Checkbox Skill

This is a **rigid** process skill. Follow every step exactly — no shortcuts, no "PASS" assertions without evidence.

## Verify behavior, not implementation

This is the single most important principle in this skill. Read it twice.

**Grepping a diff is not verification.** Checking that a line exists in a diff proves someone typed it — it does not prove the system works. A config file can have the right content and still be ignored. A workflow can have the right YAML and still fail at runtime. A dependency can be listed and still not install.

When verifying a PR, you must test what the change **does**, not what it **says**. Every verification step should answer: "does the system behave correctly after this change?" — not "does the diff contain the expected string?"

### The hierarchy of verification (strongest to weakest)

1. **Run the actual tool and observe output** — `pyright --project pyrightconfig.json` proves type checking works. `pytest -n auto` proves parallel execution works. `python3 -c "import yaml; yaml.safe_load(open('codecov.yml'))"` proves YAML is valid. This is real verification.

2. **Query the live system** — `gh pr view --json labels` to confirm metadata is set. `gh api repos/.../contents/FILE` to confirm a file exists on the branch. These test actual state, not diff content.

3. **Parse and validate file content** — Read the actual file (not the diff), parse it programmatically, and assert properties. `python3 -c "import json; c=json.load(open('config.json')); assert c['mode'] == 'basic'"` is better than `grep 'basic' config.json`.

4. **Grep the diff** — This is the weakest form. It only confirms a string is present in the change. Use this ONLY for "does the PR body contain `Closes #N`" checks or when no stronger method is possible. Never use `gh pr diff | grep` as the sole verification for functional behavior.

### Examples: bad vs. good

**Bad** (verifying implementation):
```bash
$ gh pr diff 191 | grep 'threshold: 1%'
+        threshold: 1% # fail if coverage drops by more than 1%
```
This proves someone wrote "1%" in the diff. It doesn't prove codecov will enforce it.

**Good** (verifying behavior):
```bash
$ python3 -c "
import yaml
cfg = yaml.safe_load(open('.github/codecov.yml'))
threshold = cfg['coverage']['status']['project']['default']['threshold']
print(f'Project threshold: {threshold}')
assert threshold == '1%', f'Expected 1%, got {threshold}'
"
Project threshold: 1%
```
This parses the actual file and asserts the value programmatically.

**Bad** (verifying implementation):
```bash
$ gh pr diff 195 | grep 'typeCheckingMode'
+  "typeCheckingMode": "basic",
```

**Good** (verifying behavior):
```bash
$ pyright --project pyrightconfig.json 2>&1 | tail -1
0 errors, 0 warnings, 0 informations
```
This runs the actual type checker with the actual config and proves it works.

### When to check out the PR branch

To verify behavior, you often need the PR's code locally. Check out the branch or use an existing worktree:

```bash
# Option 1: Use existing worktree if available
cd /path/to/worktree

# Option 2: Create a temporary worktree
git worktree add /tmp/verify-pr-NNN origin/branch-name
cd /tmp/verify-pr-NNN
# ... run verification ...
git worktree remove /tmp/verify-pr-NNN
```

Diff-grepping should be reserved for metadata checks (labels, milestones, issue references) where the "behavior" IS the text content.

## Why this matters

Verification without evidence is just opinion. When someone reads a PR months later, they need to see exactly what was run and what came back — not a table of "PASS" labels. Checkboxes with commands and output make verification auditable, reproducible, and trustworthy.

## The Rule

Every verification step gets a checkbox. Every checkbox shows the command and its output. The checkbox is only ticked if the result is unambiguous.

## Process

### Step 1: Run each verification step and capture output

For every verification step defined in the PR:

1. Check out the PR branch or use an existing worktree — you need the actual files, not just the diff
2. Run commands that test **behavior** (run the tool, parse the file, execute the config)
3. Capture the full console output
4. Determine: does this **unambiguously** pass, fail, or is it ambiguous?

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

- **Grepping the diff as your primary verification** — `gh pr diff | grep 'expected line'` proves a line was typed, not that it works. Always prefer running the tool, parsing the file, or querying live state.
- Writing a table with "PASS" / "FAIL" columns but no commands or output
- Ticking `[x]` when output contains warnings you haven't investigated
- Saying "all checks pass" without showing what was run
- Posting verification results without the actual console output
- Summarizing output as "looks good" instead of showing it
- Using `gh pr diff | grep` for anything other than metadata checks (issue refs, PR body content)
