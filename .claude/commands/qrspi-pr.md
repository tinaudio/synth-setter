---
description: QRSPI PR — open the PR, drive CI green, and resolve every inline comment
argument-hint: [pr-number]
---

Invoke the `tinaudio-synth-setter-skills:qrspi-pr` skill via the Skill tool,
passing `$ARGUMENTS` through unchanged (empty is fine — the skill autodetects
the PR from the current branch). That skill carries the canonical QRSPI PR
implementation; this file is a short-name alias.
