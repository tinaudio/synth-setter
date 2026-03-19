# GitHub Metadata Taxonomy

> **Status**: Draft
> **Author**: ktinubu@
> **Last Updated**: 2026-03-19

______________________________________________________________________

### Index

| §   | Section                                                    | What it covers                                    |
| --- | ---------------------------------------------------------- | ------------------------------------------------- |
| 1   | [Overview](#1-overview)                                    | How GitHub metadata organizes work in this repo   |
| 2   | [Issue Types](#2-issue-types)                              | Native Epic, Phase, Task, Bug, Feature            |
| 3   | [Hierarchy](#3-hierarchy)                                  | Epic → Phase → Task via native sub-issues         |
| 4   | [Blocking & Dependencies](#4-blocking--dependencies)       | Native blocking, file-overlap sequencing          |
| 5   | [Priority](#5-priority)                                    | Priority via project fields                       |
| 6   | [Labels](#6-labels)                                        | Domain labels for work stream classification      |
| 7   | [Milestones](#7-milestones)                                | Milestones mapping to product releases            |
| 8   | [Projects](#8-projects)                                    | Org-level GitHub Projects V2, views, status       |
| 9   | [Design Doc ↔ Issue Linkage](#9-design-doc--issue-linkage) | How design docs reference and track GitHub issues |
| 10  | [Schema](#10-schema)                                       | Entity-relationship model                         |
| 11  | [Issue Lifecycle](#11-issue-lifecycle)                     | How an issue moves from creation to close         |
| 12  | [Changes Required](#12-changes-required)                   | Migration steps and setup for native features     |

______________________________________________________________________

## 1. Overview

synth-setter organizes work using GitHub's native issue tracking: **Issue Types** (Epic, Phase, Task, Bug, Feature), **native blocking**, **sub-issues** for hierarchy, **Projects V2** for views, and **milestones** for releases. Labels provide domain classification only.

Each work stream follows: **design doc → Epic → Phases → Tasks**.

## 2. Issue Types

Issues are classified using GitHub's native Issue Types (org-level):

| Type        | Purpose                                          | Example                                |
| ----------- | ------------------------------------------------ | -------------------------------------- |
| **Epic**    | Umbrella issue grouping phases for a work stream | #74 distributed data pipeline          |
| **Phase**   | Large feature area within an epic                | #69 Pipeline Core                      |
| **Task**    | Unit of work (standalone or within a phase)      | #102 storage layer, CI improvements    |
| **Bug**     | Something isn't working                          | #10 OmegaConf resolver re-registration |
| **Feature** | A request, idea, or new functionality            | #23 improve generation throughput      |

Types are set on the issue itself, filterable in issue lists, and show distinct icons.

**Hierarchy types** (Epic, Phase) define structure. **Work types** (Task, Bug, Feature) are orthogonal — a Task under a Phase is what was previously called a "step".

**Hierarchy types** (Epic, Phase) define structure. **Work types** (Task, Bug, Feature) are orthogonal — a Task under a Phase is what was previously called a "step".

## 3. Hierarchy

All work streams use the same structure:

```
Epic (type: Epic)
├── Phase 1 (type: Phase, sub-issue of Epic)
│   ├── Task 1.1 (type: Task, sub-issue of Phase 1)
│   └── Task 1.2
├── Phase 2
│   └── Task 2.1
└── Phase N
```

- **Phase** — a large feature or functional area. Each phase is a sub-issue of its epic.
- **Task** — a testable unit of work. A task under a phase is what was previously called a "step". Standalone tasks exist outside the hierarchy.
- **PR** — a shipping unit, orthogonal to the hierarchy. A PR may contain one task, multiple tasks, or part of a large task.

Hierarchy is tracked via native sub-issues (up to 8 levels). Projects render this as an expandable tree via **hierarchy view**.

### Naming

- Phases: `Phase N: Name` (e.g., "Phase 2: Pipeline Core")
- Tasks under phases: `Task N.M: Name` (e.g., "Task 2.1: Schemas")

### Merge path

All PRs merge to `main`. Phase ordering defines the dependency chain, but PRs within a phase can land in any order as long as tasks are independently testable.

### Current epics

| Epic | Title                                                      | Project         | Design Doc                           |
| ---- | ---------------------------------------------------------- | --------------- | ------------------------------------ |
| #74  | feat(pipeline): distributed data pipeline                  | Data Pipeline   | `data-pipeline.md`                   |
| #98  | feat(eval): evaluation pipeline — predict, render, metrics | Evaluation      | `eval-pipeline.md` (PR #101)         |
| #99  | feat(storage): R2 integration for datasets and checkpoints | Eval + Pipeline | `eval-pipeline.md` §6                |
| #107 | feat(training): training pipeline & ops                    | Training        | `training-ops-braindump.md` (PR #84) |

## 4. Blocking & Dependencies

Blocking is tracked via GitHub's **native dependency system**:

- **Mark as blocked by** / **Mark as blocking** — set from the issue sidebar under "Relationships"
- Blocked issues show a **Blocked icon** in project boards and issue lists
- Dependencies are machine-readable via the GraphQL API (`blockedBy`, `blocking` fields)
- **File-overlap sequencing** — within the same work stream, tasks that modify the same files should be sequenced to avoid merge conflicts

### Critical paths

Each work stream's design doc defines a dependency DAG:

- **Data pipeline:** Phase 1 → Phase 2 → {Phase 3, Phase 4} → Phase 5 → Phase 6
- **Eval pipeline:** #94 → #85 → #88 → #89

For detailed blocking matrices and parallel execution windows, see the respective design docs.

## 5. Priority

Priority is tracked via a **Priority** single-select field on each project:

| Priority | Typical usage                          |
| -------- | -------------------------------------- |
| P0       | Critical                               |
| P1       | Foundation phases, core stages, rclone |
| P2       | Docker, E2E, production, consolidation |
| P3       | Nice-to-have                           |

## 6. Labels

Labels classify issues by **domain** only. Type and blocking are handled by native features; priority is a project field.

| Label           | Color   | Description                                   | Project |
| --------------- | ------- | --------------------------------------------- | ------- |
| `data-pipeline` | #0e8a16 | Data Pipeline project                         | #2      |
| `ci-automation` | #1d76db | CI & Automation project                       | #1      |
| `code-health`   | #fbca04 | Code Health project                           | #3      |
| `evaluation`    | #C5DEF5 | Evaluation pipeline, metrics, and inference   | #4      |
| `testing`       | #0E8A16 | Test infrastructure, fixtures, CI test config | #1      |
| `training`      | #8B5CF6 | Training pipeline, ops, and infrastructure    | #5      |

Workflow labels (`duplicate`, `invalid`, `wontfix`, `good first issue`, `help wanted`, `question`) are retained for their standard GitHub purposes.

## 7. Milestones

| Milestone            | Work Stream     |
| -------------------- | --------------- |
| data-pipeline v1.0.0 | Data Pipeline   |
| evaluation v1.0.0    | Evaluation      |
| training v1.0.0      | Training        |
| ci-automation v1.0.0 | CI & Automation |
| code-health v1.0.0   | Code Health     |

Every work stream has a milestone. Sub-issues automatically inherit their parent's milestone.

## 8. Projects

Org-level GitHub Projects V2, linked to the repo:

| #   | Project         | Custom Fields                     |
| --- | --------------- | --------------------------------- |
| 1   | CI & Automation | Priority, Start Date, Target Date |
| 2   | Data Pipeline   | Priority, Start Date, Target Date |
| 3   | Code Health     | Priority, Start Date, Target Date |
| 4   | Evaluation      | Priority, Start Date, Target Date |
| 5   | Training        | Priority                          |

### Built-in fields (all projects)

Title, Assignees, Status (`Todo` → `In Progress` → `Done`), Labels, Linked PRs, Milestone, Repository, Reviewers, Parent issue, Sub-issues progress.

### Views

- **Table** — flat list, sortable/filterable by any field
- **Board** — kanban grouped by Status
- **Roadmap** — timeline using Start Date / Target Date
- **Hierarchy** — expandable Epic → Phase → Task tree (up to 8 levels)

### Cross-project issue sharing

Some issues appear in multiple projects for cross-cutting visibility (e.g., R2 integration issues in both Data Pipeline and Evaluation). GitHub Projects V2 shares a single status field across projects, so status drift is not a concern.

## 9. Design Doc ↔ Issue Linkage

Design docs and GitHub issues are cross-referenced through these conventions:

| Convention                | Example                                                    |
| ------------------------- | ---------------------------------------------------------- |
| Design doc header         | `> **Tracking**: #98 (eval epic), #99 (R2 epic)`           |
| Implementation plan index | `§5 Phase 1 — Foundation → #68`                            |
| Issue body reference      | `**Design doc:** data-pipeline.md §7.1 (Storage as truth)` |
| Completion tracking       | `### Step 1.1 (#78) ✅ — Completed in PR #75.`             |

Design docs also include dependency graphs, blocking matrices, and timeline visualizations.

## 10. Schema

```
ISSUE
  ├── has one → ISSUE_TYPE (Epic | Phase | Task | Bug | Feature)
  ├── has one → PRIORITY (P0 | P1 | P2 | P3) — via project field
  ├── has one → MILESTONE
  ├── has one → PROJECT (via project membership)
  ├── has many → LABELS (domain only)
  ├── has one → PARENT ISSUE (native sub-issue)
  ├── has many → SUB-ISSUES
  ├── blocks / blocked by → other ISSUEs (native dependencies)
  ├── linked to → PR
  └── tracked in → DESIGN_DOC
```

## 11. Issue Lifecycle

1. Set the **Issue Type** (Epic, Phase, Task, Bug, Feature)
2. Add **domain label** (data-pipeline, evaluation, etc.)
3. Assign to a **milestone**
4. Add to the relevant **project** (Status: **Todo**)
5. Set **Priority** via the project field
6. If blocked, add **native blocking** dependency via the sidebar
7. When work starts, move to **In Progress**
8. Link the PR
9. When the PR merges, move to **Done** and close the issue

## 12. Changes Required

### Org migration

See `docs/org-migration-checklist.md` (PR #116) for the full pre/during/post checklist. Key steps:

- Create GitHub org and transfer repo
- Re-create repo secrets (ANTHROPIC_API_KEY, APPROVAL_BOT_APP_ID, APPROVAL_BOT_PRIVATE_KEY, RUNPOD_API_KEY)
- Verify Projects V2 still linked

### Native features to enable post-migration

| Feature             | What to set up                                                                | What it replaces                                     |
| ------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Issue Types**     | Create Epic and Phase types in org settings (Task, Bug, Feature are defaults) | Title-prefix naming conventions, `enhancement` label |
| **Native blocking** | Add `blockedBy`/`blocking` relationships on existing issues via sidebar       | `blocked` label, `## Blocked by` body text           |
| **Hierarchy view**  | Enable in project table views                                                 | Manual expand/collapse                               |

### Cleanup (run after native features are set up)

**1. Migrate blocking relationships to native**, then remove `blocked` label and `## Blocked by` body text from issues:

```bash
# Add native dependency
BLOCKED=$(gh issue view <blocked_num> --json id -q .id)
BLOCKER=$(gh issue view <blocker_num> --json id -q .id)
gh api graphql -f query="
mutation {
  addBlockedBy(input: {
    issueId: \"$BLOCKED\"
    blockingIssueId: \"$BLOCKER\"
  }) { blockedIssue { number } }
}"
```

**2. Delete retired labels** (`bug` kept — used by Dependabot):

```bash
gh label delete "enhancement" --yes
gh label delete "blocked" --yes
gh label delete "P0 🔴" --yes
gh label delete "P1 🟠" --yes
gh label delete "P2 🟡" --yes
gh label delete "P3 🔵" --yes
```

**3. Delete retired project fields:**

```bash
# Phase field — Data Pipeline and Evaluation
for p in 2 4; do
  gh project field-list $p --owner <org> --format json \
    | jq -r '.fields[] | select(.name == "Phase") | .id' \
    | xargs -I{} gh project field-delete --id {}
done
```

**4. Set issue types on existing issues** — assign Epic, Phase, Task, Bug, Feature types to all open issues via the sidebar or GraphQL `updateIssue` mutation.
