---
name: pr-checkbox
description: Rigid process skill for posting PR verification results. Use this skill whenever running verification steps on a PR, posting validation results to a PR, or claiming that checks pass. Also trigger when writing verification sections in PR descriptions, commenting verification outcomes, or updating verification checkboxes. If you are about to write "PASS" or "FAIL" or a verification table on a PR, you MUST use this skill first. Trigger on any PR review, validation, or verification activity.
---

# PR Verification Checkbox Skill

This is a **rigid** process skill. No shortcuts. No "PASS" assertions without evidence.

## VERIFY BEHAVIOR, NOT IMPLEMENTATION

This is the foundation of this entire skill. Everything else is formatting. If you get this wrong, perfect checkboxes are worthless.

### What this means

Treat the system as a **black box**. You put something in, you check what comes out. You do not care how the inside works. You care whether the *promise* was kept.

**The restaurant test:** You order a medium-rare steak. When the plate arrives, you check if the steak is medium-rare. You do NOT go into the kitchen and check if the chef used a cast-iron skillet, flipped it three times, or seasoned it with kosher salt. If the chef switches to a better grill tomorrow, the steak is still medium-rare — your test should still pass.

**The contract:** Every change makes a promise. "This config will make pyright check types in basic mode." "This workflow will run tests every night at 6am UTC." "This Makefile target will run tests in parallel." The verification tests the *promise*, not the *wiring* that fulfills it.

**The litmus test:** If someone rewrote the internals from scratch but kept the same inputs and outputs, would your verification still pass? If not, you're testing implementation.

### The hierarchy (strongest → weakest)

| Level | What you do | Example | Tests |
|-------|------------|---------|-------|
| **1. Exercise the code path** | Actually invoke the tool/system and observe the result | `pyright --project pyrightconfig.json` | Does type checking actually work? |
| **2. Trigger the behavior end-to-end** | Create a real input and check the real output | Create an issue with `gh issue create`, then verify it has correct metadata | Does the workflow actually produce the right result? |
| **3. Parse the actual file and assert properties** | Read the deployed file, parse it, assert | `python3 -c "import yaml; cfg=yaml.safe_load(open('codecov.yml')); assert cfg[...] == '1%'"` | Does the file contain valid, correct values? |
| **4. Query live system state** | Ask GitHub/CI what the current state is | `gh pr view --json labels` | Is the metadata actually set? |
| **5. Grep the diff** | Check if a string exists in the change | `gh pr diff \| grep 'Closes #123'` | Was a specific string typed? |

**Level 5 is almost always wrong for functional checks.** It only confirms someone typed a string. Use it exclusively for PR body content checks ("does it reference the issue?") where the text IS the behavior.

### Concrete DO / DON'T

```
DO:   Run the tool         →  pyright --project pyrightconfig.json
DO:   Exercise the path    →  gh issue create ... && gh api .../issues/N to verify metadata
DO:   Assert on output     →  assert result.exit_code == 0
DO:   Check side effects   →  ls output_file.json && python3 -c "assert valid"

DON'T: Grep the diff       →  gh pr diff | grep 'typeCheckingMode'
DON'T: Check internal state →  assert len(obj._internal_list) == 3
DON'T: Verify wiring       →  "did the function call parseInput()?"
DON'T: Read instead of run  →  parse config to check a value when you could run the tool
```

### If verification steps test implementation, rewrite them

When you encounter verification steps (in a PR description or issue) that test implementation instead of behavior, **rewrite them**. Show what you changed:

```markdown
- [x] **Pyright config works** *(rewritten: original checked `gh pr diff | grep typeCheckingMode` → now exercises pyright directly)*
  ```bash
  $ pyright --project pyrightconfig.json 2>&1 | tail -1
  0 errors, 0 warnings, 0 informations
  ```
```

Always include `(rewritten: original checked X → now checks Y)` so reviewers can see what changed and why.

### Spend time understanding what each check is meant to prove

Before writing a verification command, ask: **what is this check actually trying to prove?**

- "CODEOWNERS exists" → The real question is: will GitHub auto-assign reviewers? Exercise it: push a change to a covered path and check if a reviewer is requested, or at minimum verify GitHub's API recognizes the file on the branch.
- "pytest-xdist in requirements" → The real question is: do tests run in parallel? Exercise it: run `pytest -n auto` and check for `[gw0]` worker prefixes in the output.
- "Nightly workflow has cron trigger" → The real question is: will this workflow fire on schedule? Exercise it: trigger the workflow manually via `gh workflow run` and check it starts, or at minimum validate the cron expression with a parser.
- "Skill has frontmatter" → The real question is: does the skill trigger and produce the right output? Exercise it: invoke the skill with a test input and check the output shape.

## The Rule

Every verification step gets a checkbox. Every checkbox shows the command run and its console output. The checkbox is only ticked if the result is unambiguous.

## Process

### Step 1: Check out the PR branch

You need the actual files to exercise actual behavior. Diff-grepping is not acceptable.

```bash
git worktree add /tmp/verify-pr-NNN origin/branch-name
cd /tmp/verify-pr-NNN
```

### Step 2: Design behavioral checks

For each verification step, ask: "what promise does this change make?" Then design a command that tests whether the promise was kept by exercising the actual code path.

If the original verification steps test implementation (grep the diff, check if a line is present), **rewrite them** to test behavior. Show the before → after.

### Step 3: Run and capture output

Run each command and capture the full console output. Every check must show:
1. The exact command run (prefixed with `$`)
2. The actual console output
3. Your determination: unambiguous pass, fail, or ambiguous

### Step 4: Format with checkboxes

Size-based placement:

#### Small output (< ~20 lines) — inline

```markdown
- [x] **Pyright runs clean with this config**
  ```bash
  $ pyright --project pyrightconfig.json 2>&1 | tail -1
  0 errors, 0 warnings, 0 informations
  ```
```

#### Medium output (20-100 lines) — PR comment with link

```markdown
- [x] **Tests run in parallel** — [verification output](#issuecomment-12345)
```

#### Large output (100+ lines) — Gist with summary

```markdown
- [x] **Full test suite passes** — [summary + gist](#issuecomment-67890)
```

### Step 5: Tick or leave unchecked

- **`[x]`** — Output unambiguously confirms the promise was kept. Zero doubt.
- **`[ ]`** — Any ambiguity. Explain what's unclear.

A false `[x]` is worse than a cautious `[ ]`.

### Step 6: Post to the PR

Post as a comment with header `## Verification Results`.

## Anti-patterns

- **Grepping the diff as verification** — `gh pr diff | grep 'expected line'` proves a line was typed. It does not prove the system works. This is the #1 failure mode.
- **Parsing a file when you could run the tool** — If you can exercise the code path, do that instead of reading the config. Running `pyright` beats reading `pyrightconfig.json`.
- **Checking that a file exists instead of checking that it works** — `ls SKILL.md` proves a file is there. Invoking the skill proves it works.
- **Writing PASS/FAIL tables without commands or output**
- **Ticking `[x]` when output contains warnings you haven't investigated**
- **Saying "all checks pass" without showing evidence**

## Tests should be documentation of the requirements, not a transcript of the code.
