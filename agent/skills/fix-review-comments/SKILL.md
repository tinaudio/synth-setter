---
name: fix-review-comments
description: |-
  Apply every comment-hygiene finding from the pre-PR review sentinel, then
  refresh the sentinel so the gate passes. Use this after
  `/repo-review-full-no-comments` reports comment-hygiene BLOCK/WARN findings,
  whenever `gh pr create` is blocked with "unresolved comment-hygiene
  finding(s)", or any time you want the comment/docstring findings in the
  latest review applied in one pass. Reads the sentinel written by
  `agent/_shared/review_sentinel.py`; pairs with the `comment-hygiene` skill
  for the checklist and rewrite rules.
---

# fix-review-comments — Apply Comment-Hygiene Findings From the Sentinel

`/repo-review-full-no-comments` writes a rendered review to a sentinel file
keyed by HEAD's SHA. The `pre-pr-review-gate.sh` hook blocks `gh pr create`
while that sentinel still lists `[comment-hygiene:warn]` / `[comment-hygiene:block]`
findings (`REVIEW_COMMENT_GATE=block`, the default). This skill drains those
findings: apply each rewrite, commit, then re-review so the refreshed sentinel
is clean and the gate opens.

The fix is an **apply step, not a re-judgment**. Consistency comes from
applying the rewrite the review already produced — not from re-deriving one.
You MUST complete every step in order.

## Step 1: Locate and read the sentinel

```bash
REVIEW_PATH=$(python3 agent/_shared/review_sentinel.py path "$(git rev-parse HEAD)")
test -f "$REVIEW_PATH" || echo "MISSING: $REVIEW_PATH"
```

If the file is missing, the review hasn't run against this HEAD. Stop and tell
the user to run `/repo-review-full-no-comments` first — do not fabricate one.

Read the whole sentinel. The findings you act on are the lines tagged
`**[comment-hygiene:warn]**` or `**[comment-hygiene:block]**`, grouped under
their `### \`<path>\`\` headers. Ignore every other skill's findings — they are
out of scope here.

## Step 2: Extract one work-item per comment-hygiene finding

For each tagged finding, capture:

- `path` and `line` (from the `**L<line>**` anchor and its `### \`<path>\`\` header).
- The **C-id** (`C1`–`C12`) named in the description.
- The verbatim `Before:` / `After:` rewrite **if the body carried one**. The
  review preserves code fences, so most findings ship their own rewrite.

When a finding's body is terse and carries no usable `After:`, read the cited
`path:line` (±2 lines) and derive the rewrite straight from the
`comment-hygiene` checklist for that C-id. This is the only place judgment
re-enters; keep it mechanical — apply the rule, don't re-litigate whether the
finding is valid.

## Step 3: Bucket each finding — apply now vs. surface

| Bucket            | C-ids                       | Action                                                                         |
| ----------------- | --------------------------- | ------------------------------------------------------------------------------ |
| **Mechanical**    | C2, C3, C6, C7, C8, C9, C11 | Apply the rewrite (deletion or tightening) verbatim.                           |
| **Move-in-place** | C1                          | Lift the comment above the `- name:` step. Surface if the move is non-obvious. |
| **Needs a ref**   | C5, C10                     | Apply only when a real issue/rationale is available (Step 4). Else surface.    |
| **Rewrite-ship**  | C12                         | Apply the `After:` if present; else tighten per the doc-map rules.             |

The **mitigation that keeps the hard gate honest**: never invent a placeholder
issue number to satisfy C5/C10. A fake `# … — see #0` would pass both the
checklist pattern and the gate while baking wrong content into the code. If you
can't source a real ref, leave that one finding unfixed and report it — the
gate legitimately stays closed until a human supplies it (or sets
`REVIEW_COMMENT_GATE=off` for a genuinely intentional finding).

## Step 4: Source a real issue ref for C5 / C10 (mitigation)

Before surfacing a needs-a-ref finding, try to find a real number already tied
to this work:

```bash
gh pr view --json body -q .body 2>/dev/null | grep -oE '(Refs|Closes|Fixes|Part of) #[0-9]+'
git log -1 --pretty=%B | grep -oE '#[0-9]+'
```

Use a number only if it actually governs this change (the PR's linked issue or
the commit's trailer). If nothing applies, do not guess — surface the finding.

## Step 5: Apply the fixes — content-anchored, bottom-up

Findings are anchored to `path:line`, so editing top-down shifts every later
line in the same file. Apply **per file, in descending line order**, and match
on the verbatim `Before:` text (exact-match `Edit`), not on the line number.
Content-anchoring survives any drift the review's line numbers may already
have.

Do not touch code the findings don't cite. This skill changes inline text
only — comments, docstrings, YAML comments, `doc-map.yaml` prose.

## Step 6: Commit the fixes

The re-review in Step 7 reads committed history (`base..HEAD`), so uncommitted
edits won't clear the findings. Commit the applied fixes:

```bash
git add -A
git commit -m "chore(comments): apply comment-hygiene fixes from pre-PR review"
```

Conventional-commit + gitlint rules apply (see `/github-taxonomy` for scope).
If these fixes belong with an unpushed WIP commit, squash them in instead of
adding a separate commit — your call based on whether the prior commit is
already pushed.

## Step 7: Re-review to refresh the sentinel

Run `/repo-review-full-no-comments` again. It writes a fresh sentinel against
the new HEAD. This both regenerates the gate's contract at the current SHA and
re-reviews your fixes (a bad rewrite gets re-flagged here rather than shipping).

## Step 8: Report

Summarize:

- **Applied** — count by C-id, with `path:line` for each.
- **Surfaced** — every finding left unfixed and why (needs a real issue ref, a
  rationale decision, or a non-obvious C1 move). These are what the user must
  resolve before the gate opens.
- **Gate status** — whether the refreshed sentinel is clean. If findings
  remain, name the `gh pr create … # REVIEW_FULL=<path>` follow-up, and note
  that `REVIEW_COMMENT_GATE=off` bypasses the sub-gate for an intentional
  finding the user has accepted.

## Notes

- This skill is the remediation half of the pre-PR comment gate; the detection
  half is `/repo-review-full-no-comments` + `pre-pr-review-gate.sh`. Keep the
  three in sync — the gate greps for the `[comment-hygiene:<severity>]` tag the
  review emits.
- Rewrite semantics (what each C-id means, the `.. attribute ::` and
  no-`:param self:` pydoclint rules) live in the `comment-hygiene` skill. When
  re-deriving a fix in Step 2, that skill is authoritative — do not reinvent the
  rules here.
- Re-running is cheap relative to shipping wrong content. If a fix looks
  ambiguous, surface it rather than guessing.
