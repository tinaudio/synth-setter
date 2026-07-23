---
name: correctness-review
description: |-
  Adversarial correctness and bug review of a PR diff — hunt for logic errors,
  off-by-one and boundary mistakes, inverted conditions, unhandled None/empty,
  wrong operators, resource/exception mishandling, concurrency races, and
  silently-wrong numeric/dtype/shape handling. Conservative by design: a genuine
  correctness defect BLOCKS the merge, and when a reviewer is unsure whether a
  code path is correct it flags rather than waves it through. Every finding
  carries a concrete failure scenario (inputs → wrong result). Used by the
  `/repo-review-full` and `/repo-review-full-no-comments` fan-out; repo-local,
  invoked bare (no `tinaudio-synth-setter-skills:` prefix).
---

# correctness-review — Adversarial Bug & Correctness Review

Find the bugs. This skill's only job is to decide, for the changed code, **will
it do the wrong thing for some reachable input or state?** It does not review
style, naming, docstrings, or structure — `code-health`, `python-style`, and
`comment-hygiene` own those. It reasons about *behavior*: what the code computes
versus what it must compute.

## Posture: conservative, blocks for issues

Correctness is not negotiable, so this reviewer is deliberately biased toward
blocking:

- **A real correctness defect is always `BLOCK`** — never downgrade a wrong
  result, a crash on a reachable input, or a broken invariant to `WARN` because
  it "seems unlikely" or "is probably fine."
- **When you are unsure whether a path is correct, flag it.** Uncertainty about
  behavior is itself the signal. If you cannot convince yourself a branch is
  correct from the diff and the surrounding code, raise it rather than assume
  the author got it right.
- **`WARN` is only for latent or lower-confidence concerns** — a bug that needs
  an input the code doesn't currently receive, a fragile assumption that holds
  today but will break under a plausible near-future change, or a defect you
  can describe but not yet tie to a concretely reachable trigger.

The one discipline that keeps a block-heavy reviewer trustworthy: **every
finding must name a concrete failure scenario** — specific inputs or state that
reach the defect, and the wrong result that follows. "This looks risky" without
a trigger is not a finding; drop it or turn it into a WARN with the missing-link
stated. A blocked PR must always come with the case that justifies the block.

## Step 1: Read the diff and its blast radius

Get the changed code and read enough of the surrounding code to judge it —
callers, callees, and the invariants they rely on. `$BASE` and `$HEAD` are the
PR's base- and head-commit SHAs (from `gh api repos/<owner>/<repo>/pulls/<n> --jq .base.sha`
and `gh pr view <N> --json headRefOid`, or as set by the harness).

```bash
git diff --diff-filter=d "$BASE"..."$HEAD"
```

Read every changed file at `$HEAD` with the `Read` tool (not `cat`). For each
changed function, pull up its callers and the functions it calls — a diff is
correct or not only relative to the contract its neighbors assume. Skim the PR
description for the *intended* behavior; a change that is internally consistent
but does the wrong thing is still a bug.

## Step 2: Hunt — the correctness checklist

Walk every changed code path against these classes. For each suspected defect,
construct the triggering input before writing it up.

**Control flow & conditions**

- Inverted or off-by-one conditions (`<` vs `<=`, `>` vs `>=`, `and` vs `or`,
  a negation dropped or doubled).
- Boundary values: empty collection, single element, first/last index, zero,
  negative, the max, an exactly-equal case.
- Missing `else` / unhandled branch — a case that falls through to the wrong
  default or to no action at all.
- Early `return`/`break`/`continue` that skips required cleanup or leaves state
  half-updated.

**Data & state**

- Unhandled `None`/`null`/empty/missing-key — a value that can be absent reaching
  code that assumes it is present (attribute access, indexing, arithmetic).
- Mutated shared or default-argument state that leaks across calls/iterations.
- Aliasing: two names bound to the same mutable object where the code assumes
  independent copies.
- Stale reads: a value captured before a mutation and used as if fresh.

**Numeric, dtype & shape (pipeline-critical here)**

- Integer vs float division, silent `float64`↔`float32` drift, overflow/underflow,
  precision loss where an exact value is required.
- Array shape/axis mismatches, broadcasting that silently produces the wrong
  result instead of erroring, wrong axis in a reduction.
- Off-by-one in slicing/windowing/framing; sample-rate or bin-count assumptions
  that don't match the spec.

**Errors, resources & concurrency**

- An exception path that leaves a file/lock/handle open or a partial write
  committed (relate to the `.valid`-marker commit-point invariant when shards
  are involved).
- Swallowed exceptions that mask a failure and let wrong data flow downstream.
- Return values / error codes ignored where failure must stop the pipeline.
- Race conditions and non-atomic read-modify-write on state shared across
  workers/threads/processes.
- Ordering assumptions on dict/set/async completion that don't hold.

**Contract & regression**

- The change breaks an invariant a caller depends on (return type now nullable,
  units changed, ordering no longer guaranteed, a raised exception replaced by a
  silent sentinel).
- A fixed bug with no test pinning it, or a code path the diff makes reachable
  for the first time with no coverage — note it (defer deep test-adequacy
  judgment to `tdd-implementation`/`ml-test`, but flag a correctness gap they
  would miss).

This list is a prompt, not a cage — a bug that fits none of these categories is
still a bug. Report it.

## Step 3: Verify each finding before you write it

For every candidate defect, do the adversarial check:

1. Name the **exact input or state** that reaches the defect.
2. Trace what the code **actually** does with it, line by line.
3. State the **wrong result** (crash, silent bad value, broken invariant) and,
   in one clause, the **correct** behavior.
4. Confirm the path is **reachable** — if you cannot find a caller or state that
   reaches it today, it is a `WARN` (latent), not a `BLOCK`.

If the trace shows the code is actually correct, drop the candidate — a
false-positive BLOCK erodes the trust that makes a conservative reviewer
useful. Being conservative means flagging genuine uncertainty, not inventing
triggers that don't exist.

## Step 4: Output — the fan-out report contract

Return the standard fan-out report. **Severity:**

- **BLOCK** — a reachable input/state produces a wrong result, a crash, or a
  broken invariant. All genuine correctness defects land here.
- **WARN** — latent or lower-confidence: needs input the code doesn't yet
  receive, a fragile assumption that will break under a plausible change, or a
  defect you can describe but not tie to a concrete trigger.

Every finding description MUST contain, in this order:

1. One sentence naming the defect (the anchor lives in `path`/`line`).
2. The **failure scenario** — concrete inputs/state → the wrong result, and the
   correct behavior in one clause.

Report shape (same JSON contract the fan-out expects — one object, no fence or
surrounding prose):

```json
{
  "skill": "correctness-review",
  "target": "PR #<N>",
  "findings": [
    {
      "severity": "block",
      "path": "<repository-relative changed path>",
      "line": 42,
      "description": "<defect>. Failure: given <input/state>, <what happens>; should <correct behavior>."
    }
  ],
  "what_looks_good": ["<a tricky path that IS correct, and why — so the next reviewer need not re-derive it>"]
}
```

Aim for a 1500-word ceiling across string values so the report stays scannable.
Rank `block` findings most-severe first in the `findings` array.

## Review checklist

| #   | Check                | Looks for                                                                    | Severity |
| --- | -------------------- | ---------------------------------------------------------------------------- | -------- |
| 1   | **Conditions**       | Inverted/off-by-one comparisons, `and`/`or` swaps, dropped negation          | BLOCK    |
| 2   | **Boundaries**       | Empty / single / first / last / zero / negative / max case mishandled        | BLOCK    |
| 3   | **Nullability**      | Reachable `None`/empty/missing-key hitting code that assumes presence        | BLOCK    |
| 4   | **State & aliasing** | Leaked mutable default, shared-object aliasing, stale-read-after-mutation    | BLOCK    |
| 5   | **Numeric/dtype**    | int/float division, `float32`↔`float64` drift, precision loss, overflow      | BLOCK    |
| 6   | **Shape/axis**       | Array shape/axis/broadcast mismatch, off-by-one slicing/windowing            | BLOCK    |
| 7   | **Error/resource**   | Exception path leaks a resource or commits a partial write; swallowed errors | BLOCK    |
| 8   | **Ignored failures** | Return code / error result discarded where failure must stop the flow        | BLOCK    |
| 9   | **Concurrency**      | Race / non-atomic read-modify-write on shared state; bad ordering assumption | BLOCK    |
| 10  | **Broken contract**  | Change violates an invariant a caller depends on (type/units/order/raises)   | BLOCK    |

Each row is a correctness-defect class and defaults to **BLOCK** — this reviewer
does not downgrade a genuine defect to WARN. Two cross-cutting disciplines from
the steps above govern how a row's severity is finally set: a defect is BLOCK
only when its trigger is **reachable today** (else WARN — latent; Step 3.4), and
**every** finding must cite a concrete failure scenario or it is dropped
(Step 2 / the Posture section). BLOCK = must fix before merge · WARN = advisory.

## Notes

- Scope: *behavior only*. Do not emit style, naming, docstring, structure, or
  Lance-native-vs-hand-rolled findings — those belong to `code-health`,
  `python-style`, `comment-hygiene`, and `lance-review`. Overlap is fine when a
  structural smell also causes a concrete wrong result; lead with the failure
  scenario, not the smell.
- The fan-out does not dedupe across skills, so don't soften a real correctness
  BLOCK just because another checklist might also touch the line — keep your
  finding's signal (the failure scenario) independent.
- Repo-local skill: invoke the bare `correctness-review` skill via the Skill
  tool; if the harness has not registered it, read and apply this file directly.
  No web access required — findings come from reasoning about the diff, not
  external docs.
