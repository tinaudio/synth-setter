---
name: pr-checkbox
description: Rigid process skill for posting PR verification results. Use this skill whenever running verification steps on a PR, posting validation results to a PR, or claiming that checks pass. Also trigger when writing verification sections in PR descriptions, commenting verification outcomes, or updating verification checkboxes. If you are about to write "PASS" or "FAIL" or a verification table on a PR, you MUST use this skill first. Trigger on any PR review, validation, or verification activity.
---

# PR Verification Checkbox Skill

This is a **rigid** process skill. No shortcuts. No "PASS" assertions without evidence.

---

## VERIFY BEHAVIOR, NOT IMPLEMENTATION

This is the foundation of this entire skill. Everything else is formatting. If you get this wrong, perfect checkboxes are worthless.

### What this means

Treat the system as a **black box**. You put something in, you check what comes out. You do not care how the inside works. You care whether the *promise* was kept.

**The restaurant test:** You order a medium-rare steak. When the plate arrives, you check if the steak is medium-rare. You do NOT go into the kitchen and check if the chef used a cast-iron skillet, flipped it three times, or seasoned it with kosher salt. If the chef switches to a better grill tomorrow, the steak is still medium-rare — your test should still pass.

**The contract:** Every change makes a promise. "This config will make pyright check types in basic mode." "This workflow will run tests every night at 6am UTC." "This Makefile target will run tests in parallel." The verification tests the *promise*, not the *wiring* that fulfills it.

**The litmus test:** If someone rewrote the internals from scratch but kept the same inputs and outputs, would your verification still pass? If not, you're testing implementation.

---

## The Hierarchy (strongest → weakest)

| Level | What you do | Example |
|-------|------------|---------|
| **1. Run the tool** | Invoke the actual tool/system, observe the result | `pyright --project pyrightconfig.json` |
| **2. Trigger end-to-end** | Create a real input, check the real output | `gh issue create ...` then verify metadata via API |
| **3. Query live system state** | Ask the live platform what it sees | `gh api repos/.../codeowners/errors` |
| **4. Parse file and assert** | Read a config, parse it, check values | `python3 -c "import yaml; ..."` |
| **5. Grep the diff** | Check if a string exists in the change | `gh pr diff \| grep '...'` |

### The escalation rule

**You MUST use the highest level available.** If Level 1 is possible, Levels 2–5 are not acceptable. If Level 2 is possible, Levels 3–5 are not acceptable. You only descend when a higher level is genuinely impossible (tool not installed, no test fixtures, external service unreachable).

When you descend, you MUST state why:

```markdown
- [x] **Config values are correct** *(Level 4 — codecov CLI not available in this environment; fell back to YAML parse)*
```

If you do not state a reason, the check is invalid.

---

## Parsing Is Not Exercising

This is the #1 way this skill gets violated. Replacing `grep` with `yaml.safe_load()` or `toml.load()` and calling it behavioral is wrong. It is the same thing in a trench coat.

**Parsing a config file proves the VALUE IS IN THE FILE. Running the tool proves THE TOOL WORKS WITH THAT VALUE.** These are different claims.

Ask yourself: **"Am I running the tool, or reading its inputs?"**

| Reading the input (NOT behavioral) | Running the tool (behavioral) |
|-------------------------------------|-------------------------------|
| `yaml.safe_load(open('codecov.yml'))` | `codecov validate codecov.yml` |
| `make -n test` (prints the recipe) | `make test 2>&1 \| grep gw` (runs the recipe) |
| Parse `.github/workflows/test.yml` | `gh workflow run test.yml && gh run watch` |
| `toml.load('pyproject.toml')` | `pytest --co -q` (proves pytest loads the config) |
| `cat CODEOWNERS` | `gh api repos/.../codeowners/errors` |
| `python3 -c "import json; ..."` on a config | Invoke whatever system consumes that config |

**If your verification command does not INVOKE the system under test, it is not behavioral. Go up the hierarchy.**

---

## Spend Time Understanding What Each Check Proves

Before writing a verification command, ask: **what is this check actually trying to prove?**

- "CODEOWNERS exists" → Real question: will GitHub auto-assign reviewers? Exercise: `gh api repos/.../codeowners/errors` returns no errors. Better: push a change and check reviewer assignment.
- "pytest-xdist in requirements" → Real question: do tests run in parallel? Exercise: run `pytest -n auto` and check for `[gw0]` worker prefixes.
- "Nightly workflow has cron trigger" → Real question: will this fire on schedule? Exercise: `gh workflow run` and check it starts.
- "Codecov threshold is 1%" → Real question: will Codecov enforce 1%? Exercise: `codecov validate codecov.yml` or trigger a coverage run.
- "Makefile has `-n auto`" → Real question: does `make test` run tests in parallel? Exercise: run `make test` and observe parallel workers.

---

## Process

### Step 1: Check out the PR branch

You need the actual files to exercise actual behavior. Diff-grepping is not acceptable.

```bash
git fetch origin <branch>
git worktree add /tmp/verify-pr-NNN origin/<branch>
cd /tmp/verify-pr-NNN
```

### Step 2: Design behavioral checks

For each verification step, ask: **"what promise does this change make?"** Then design a command that tests whether the promise was kept by **invoking the actual system**.

Start at Level 1. Only descend if Level 1 is impossible, and state why.

If the original verification steps (from a PR description, issue, or task) test implementation, **rewrite them**. Show the before → after.

### Step 3: Self-audit (MANDATORY)

Before running anything, review every check you designed and answer these three questions:

1. **Does this command INVOKE the system, or READ/PARSE a file?**
2. **If someone rewrote the internals, would this check still pass?**
3. **Is there a higher-level check I skipped because parsing felt easier?**

If any answer is "it parses a file" and a higher-level option exists, rewrite it now. Do not proceed until every check is at the highest feasible level.

**Hard gate:** If more than half your checks parse/read files instead of running tools, STOP and redesign. You are testing implementation.

### Step 4: Run and capture output

Run each command and capture the full console output. Every check must show:
1. The exact command run (prefixed with `$`)
2. The actual console output
3. Your determination: unambiguous pass, fail, or ambiguous

### Step 5: Format with checkboxes

Size-based placement:

#### Small output (< ~20 lines) — inline

```markdown
- [x] **Pyright runs clean with this config** *(Level 1 — invoked pyright directly)*
  ```bash
  $ pyright --project pyrightconfig.json 2>&1 | tail -1
  0 errors, 0 warnings, 0 informations
  ```
```

#### Medium output (20–100 lines) — PR comment with link

```markdown
- [x] **Tests run in parallel** — [verification output](#issuecomment-12345)
```

#### Large output (100+ lines) — Gist with summary

```markdown
- [x] **Full test suite passes** — [summary + gist](#issuecomment-67890)
```

### Step 6: Tick or leave unchecked

- **`[x]`** — Output unambiguously confirms the promise was kept. Zero doubt.
- **`[ ]`** — Any ambiguity. Explain what's unclear.

A false `[x]` is worse than a cautious `[ ]`.

### Step 7: Post to the PR

Post as a comment with header `## Verification Results`.

---

## Rewriting Implementation Checks

When you encounter verification steps that test implementation instead of behavior, **rewrite them**. Always include the annotation showing what changed and why:

### Example: YAML parsing → tool invocation

```markdown
BEFORE (Level 4 — parses input):
- [x] **Codecov threshold is 1%**
  $ python3 -c "import yaml; c=yaml.safe_load(open('codecov.yml'));
    assert c['coverage']['status']['project']['default']['threshold'] == '1%'"

AFTER (Level 1 — runs the tool):
- [x] **Codecov config is valid and accepted** *(rewritten: original parsed YAML asserting threshold value → now validates config through codecov CLI)*
  $ codecov validate codecov.yml
  Valid!
```

### Example: Makefile dry-run → actual execution

```markdown
BEFORE (Level 4 — reads the recipe):
- [x] **make test invokes parallel**
  $ make -n test
  pytest -n auto -m "not slow"

AFTER (Level 1 — runs the recipe):
- [x] **make test runs tests in parallel** *(rewritten: original used `make -n` dry run → now runs `make test` and checks for worker prefixes)*
  $ make test 2>&1 | head -20 | grep -E 'gw[0-9]'
  [gw0] PASSED tests/test_basic.py::test_init
  [gw1] PASSED tests/test_basic.py::test_load
```

### Example: Workflow YAML parsing → workflow trigger

```markdown
BEFORE (Level 4 — parses the workflow file):
- [x] **CI uses parallel**
  $ python3 -c "import yaml; wf=yaml.safe_load(open('.github/workflows/test.yml'));
    assert '-n auto' in wf['jobs']['test']['steps'][2]['run']"

AFTER (Level 2 — triggers the workflow):
- [x] **CI workflow runs and succeeds** *(rewritten: original parsed workflow YAML for flags → now triggers workflow and checks result)*
  $ gh workflow run test.yml --ref ci/branch && sleep 30 && gh run list -w test.yml -L 1 --json status --jq '.[0].status'
  completed
```

### Annotation format

Always use this pattern so reviewers see what changed:

```
*(rewritten: original checked X → now exercises Y)*
```

or when descending:

```
*(Level N — reason higher level is not feasible)*
```

---

## Anti-Patterns

1. **Grepping the diff as verification** — `gh pr diff | grep 'expected line'` proves a line was typed. It does not prove the system works.
2. **Parsing a config when you could run the tool** — `yaml.safe_load` is not behavioral. If the tool exists, run it.
3. **Dry-running instead of running** — `make -n test` reads the recipe. `make test` runs it. These prove different things.
4. **Checking that a file exists instead of checking that it works** — `ls SKILL.md` proves a file is there. Invoking the skill proves it works.
5. **Writing PASS/FAIL tables without commands or output**
6. **Ticking `[x]` when output contains warnings you haven't investigated**
7. **Saying "all checks pass" without showing evidence**
8. **Calling `yaml.safe_load` or `toml.load` "behavioral"** — It is not. It is reading an input, not running a system.

---

## Quick Reference

```
ALWAYS: Run the tool, trigger the behavior, observe the output.
NEVER:  Read the config, parse the file, grep the diff — unless forced.

ALWAYS: State the hierarchy level and justify descent.
NEVER:  Silently drop to Level 4 because it's easier.

ALWAYS: Show the command and its output.
NEVER:  Assert PASS without evidence.

ALWAYS: Rewrite implementation checks into behavioral checks.
NEVER:  Accept a check because "it's better than grep."
```

Tests should be documentation of the requirements, not a transcript of the code.
