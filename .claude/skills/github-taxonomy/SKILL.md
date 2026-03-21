---
name: github-taxonomy
description: Rigid process skill for all GitHub metadata operations in synth-setter. Use this skill whenever creating or updating issues, PRs, milestones, labels, priorities, blocking relationships, project fields, or any GitHub project management. Also trigger when assigning issue types (Epic, Phase, Task, Bug, Feature), setting up sub-issue hierarchy, linking PRs to issues, or managing the GitHub Project board. If the task touches GitHub metadata in any way — even tangentially — use this skill. IMPORTANT — this skill MUST also trigger whenever Claude creates a PR, to ensure a taxonomy-compliant issue is linked so the CI metadata gate passes.
---

# GitHub Taxonomy Skill

This skill enforces the project's GitHub metadata conventions. The full reference lives in `docs/design/github-taxonomy.md` — read it before any GitHub metadata operation you haven't done in this session.

## When This Triggers

- Creating or updating issues (any type)
- **Creating or updating PRs** (MUST ensure a compliant linked issue — see §Ensuring PRs Pass the Metadata Gate below)
- Setting labels, milestones, priorities, or issue types
- Establishing hierarchy (sub-issues) or blocking relationships
- Managing the GitHub Project board (adding items, setting fields, creating views)
- Any `gh` CLI or GraphQL operation that modifies GitHub metadata

## Process

This is a rigid process skill. Follow every step — do not skip steps or improvise alternatives.

### Ensuring PRs Pass the Metadata Gate

**This section is REQUIRED before every `gh pr create`.** The CI workflow `pr-metadata-gate.yaml` will fail if the PR does not link to at least one issue that has all three: issue type, domain label, and milestone.

#### Step 1: Determine the correct issue to link

Ask: does an existing issue already describe this work?

- **If yes** — use that issue number. Proceed to Step 2.
- **If no** — create a new issue (follow "Creating an Issue" below), then use its number.

#### Step 2: Verify the linked issue is taxonomy-compliant

Check all three required fields on the issue:

```bash
# Check issue type (GraphQL — not available via REST)
gh api graphql -f query='
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      issue(number: $number) {
        issueType { name }
      }
    }
  }
' -f owner="tinaudio" -f repo="synth-setter" -F number=ISSUE_NUMBER \
  --jq '.data.repository.issue.issueType.name'

# Check labels and milestone (REST)
gh api repos/tinaudio/synth-setter/issues/ISSUE_NUMBER \
  --jq '{type: "check below", labels: [.labels[].name], milestone: .milestone.title}'
```

The issue MUST have:

1. **Issue type** — one of: `Epic`, `Phase`, `Task`, `Bug`, `Feature`
2. **Domain label** — at least one of: `data-pipeline`, `ci-automation`, `code-health`, `evaluation`, `storage`, `testing`, `training`
3. **Milestone** — assigned (e.g., `data-pipeline v1.0.0`, `training v1.0.0`)

#### Step 3: Fix any missing metadata before creating the PR

If any field is missing, set it now:

```bash
# Set issue type via GraphQL (get issue node ID first)
ISSUE_NODE_ID=$(gh api graphql -f query='
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      issue(number: $number) { id }
    }
  }
' -f owner="tinaudio" -f repo="synth-setter" -F number=ISSUE_NUMBER \
  --jq '.data.repository.issue.id')

# Find the issue type ID (e.g., "Task")
ISSUE_TYPE_ID=$(gh api graphql -f query='
  query($owner: String!, $repo: String!) {
    repository(owner: $owner, name: $repo) {
      issueTypes(first: 20) {
        nodes { id name }
      }
    }
  }
' -f owner="tinaudio" -f repo="synth-setter" \
  --jq '.data.repository.issueTypes.nodes[] | select(.name == "Task") | .id')

# Set the issue type
gh api graphql -f query='
  mutation($issueId: ID!, $issueTypeId: ID!) {
    updateIssue(input: {id: $issueId, issueTypeId: $issueTypeId}) {
      issue { title }
    }
  }
' -f issueId="$ISSUE_NODE_ID" -f issueTypeId="$ISSUE_TYPE_ID"

# Add domain label
gh issue edit ISSUE_NUMBER --repo tinaudio/synth-setter --add-label "DOMAIN_LABEL"

# Set milestone
gh issue edit ISSUE_NUMBER --repo tinaudio/synth-setter --milestone "MILESTONE_NAME"
```

#### Step 4: Create the PR with the issue reference

Include the issue reference in the PR body:

- Use `Closes #N` if the PR fully resolves the issue
- Use `Refs #N` if the PR is related but doesn't fully resolve it

#### Domain-to-milestone mapping

| Domain label | Milestone |
| --- | --- |
| `data-pipeline` | `data-pipeline v1.0.0` |
| `ci-automation` | `ci-automation v1.0.0` |
| `code-health` | `code-health v1.0.0` |
| `evaluation` | `evaluation v1.0.0` |
| `storage` | `storage v1.0.0` |
| `testing` | `testing v1.0.0` |
| `training` | `training v1.0.0` |

#### Quick-create a compliant issue (when no existing issue fits)

```bash
# One-shot: create a taxonomy-compliant Task issue
gh issue create \
  --repo tinaudio/synth-setter \
  --title "feat(DOMAIN): DESCRIPTION" \
  --body "Created to track PR work." \
  --label "DOMAIN_LABEL" \
  --milestone "MILESTONE_NAME"

# Then set issue type via GraphQL (see Step 3 above)
```

### Creating an Issue

Follow the full lifecycle (taxonomy doc §11):

1. **Set Issue Type** — one of: Epic, Phase, Task, Bug, Feature
2. **Add domain label** — one of: `data-pipeline`, `ci-automation`, `code-health`, `evaluation`, `storage`, `testing`, `training`. Cross-cutting infrastructure may carry multiple domain labels (see taxonomy doc §6).
3. **Assign milestone** — use the work stream's milestone (e.g., `data-pipeline v1.0.0`)
4. **Add to the project** — set Status to `Todo`
5. **Set Priority** via the project field:
   - **P0**: Blocks all progress
   - **P1**: Needed before milestone ships
   - **P2**: Planned but not blocking
   - **P3**: Nice-to-have
6. **Set blocking** — if this issue is blocked by or blocks another, add native blocking via the sidebar
7. **Set hierarchy** — if this is a Phase, make it a sub-issue of its Epic. If this is a Task under a Phase, make it a sub-issue of that Phase.

### Naming Conventions

- Epics: `feat(<domain>): <description>` (e.g., `feat(pipeline): distributed data pipeline`)
- Phases: `Phase N: <Name>` (e.g., `Phase 2: Pipeline Core`)
- Tasks under phases: `Task N.M: <Name>` (e.g., `Task 2.1: Schemas`)
- Standalone tasks, bugs, features: use conventional commit style in the title

### Linking PRs to Issues

Use the correct linking method — never use `Fixes` or `Closes` unless the PR fully resolves the issue:

| Method | When to use | Auto-closes? |
| --- | --- | --- |
| `Fixes #N` / `Closes #N` | PR fully resolves the issue | Yes |
| `Refs #N` | PR is related but doesn't resolve | No |
| Development sidebar link | Manual non-closing connection from issue page | No |

### Milestones

Set the milestone on each issue individually. GitHub does not auto-inherit milestones from parent issues.

### Hierarchy

Use native sub-issues (not labels or text references) for all hierarchy:

```
Epic
├── Phase 1 (sub-issue of Epic)
│   ├── Task 1.1 (sub-issue of Phase 1)
│   └── Task 1.2
└── Phase 2
    └── Task 2.1
```

### Blocking & Dependencies

Use GitHub's native blocking system (issue sidebar → Relationships), not labels or text conventions. Blocked issues show a blocked icon in project boards.

### Labels

Labels are for **domain classification only**. Do not use labels for priority, blocking, or issue type — those have native features or project fields.

### Before Completing Any GitHub Operation

Verify you have set all required fields. Missing metadata creates tracking gaps that compound over time.

**Issue checklist:**
- [ ] Issue type set
- [ ] Domain label applied
- [ ] Milestone assigned
- [ ] Added to project
- [ ] Priority set
- [ ] Hierarchy established (if applicable)
- [ ] Blocking relationships set (if applicable)

**PR checklist:**
- [ ] Linked issue is taxonomy-compliant (issue type + domain label + milestone)
- [ ] Correct linking method used (`Fixes` vs `Refs`)
- [ ] Domain label applied
- [ ] Milestone assigned
- [ ] Added to project
