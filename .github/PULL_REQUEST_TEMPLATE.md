<!--
Title: conventional-commit, self-contained — name the subject, not just the action.
  e.g. "fix(storage): retry R2 puts on 5xx", not "fix bug".
Keep the canonical section order below (Why → What changed → Test plan). A CI check
(pr-metadata-gate) flags missing or misnamed sections. Delete these comments before submitting.
-->

## Why

<!-- The problem or motivation in 1-3 sentences, and the linked issue. What was wrong or missing? -->

Refs #<!-- issue number; use Closes/Fixes to auto-close, Refs/Part of for partial work -->

## What changed

<!-- The actual change as bullets. Lead with the load-bearing edit; mechanics after. -->

-

## Test plan

<!--
How a reviewer knows it is correct: commands run + outcomes, or a linked CI run.
Do NOT restate "CI green" / "ruff clean" — CI proves that automatically. Capture what CI can't:
behavior verified, manual steps, and what you deliberately did not test and why.
-->

-

## Out of scope

<!-- Deliberate non-goals and follow-ups. Delete this whole section if there are none. -->

-

______________________________________________________________________

- [ ] One logical change; any breaking changes are called out above.
