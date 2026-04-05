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
| **Epic**    | Umbrella issue grouping phases for a work stream | Epic: distributed data pipeline        |
| **Phase**   | Large feature area within an epic                | Phase 2: Pipeline Core                 |
| **Task**    | Unit of work within a phase                      | Task 2.1: Schemas                      |
| **Bug**     | Something isn't working                          | fix(ci): broken workflow               |
| **Feature** | A request, idea, or new functionality            | Feature: improve generation throughput |

Types are set on the issue itself, filterable in issue lists, and show distinct icons.

**Hierarchy types** (Epic, Phase) define structure. **Work types** (Task, Bug, Feature) are orthogonal — a Task under a Phase is what was previously called a "step".

## 3. Hierarchy

All work streams use the same structure:

```
Epic: <name>
├── Phase 1: <name> (sub-issue of Epic)
│   ├── Task 1.1: <name> (sub-issue of Phase 1)
│   └── Task 1.2: <name>
├── Phase 2: <name>
│   └── Task 2.1: <name>
└── Phase N: <name>
```

- **Phase** — a large feature or functional area. Each phase is a sub-issue of its epic.
- **Task / Bug / Feature** — work items. Every work item must trace to an epic via the sub-issue hierarchy (there are no standalone issues).
- **PR** — a shipping unit, orthogonal to the hierarchy. A PR may contain one task, multiple tasks, or part of a large task.

Hierarchy is tracked via native sub-issues (up to 8 levels). Projects render this as an expandable tree via **hierarchy view**.

### Phase-parenting rule

Task, Bug, and Feature issues MUST be sub-issues of a Phase — never direct children of an Epic. Only Phase issues should be direct children of Epics. This ensures every work item traces to an epic through exactly one phase, keeping the hierarchy view clean and enabling phase-level progress tracking.

### Naming

- Epics: `Epic: Name` (e.g., "Epic: distributed data pipeline")
- Phases: `Phase N: Name` (e.g., "Phase 2: Pipeline Core")
- Tasks under phases: `Task N.M: Name` (e.g., "Task 2.1: Schemas")
- Features: `Feature: Name` (e.g., "Feature: improve generation throughput")
- Bugs: use conventional commit style (e.g., "fix(ci): broken workflow")

### Merge path

All PRs merge to `main`. Phase ordering defines the dependency chain, but PRs within a phase can land in any order as long as tasks are independently testable.

### Current epics

| Epic | Title                                                | Domain Label    | Design Doc                           |
| ---- | ---------------------------------------------------- | --------------- | ------------------------------------ |
| #74  | Epic: distributed data pipeline                      | `data-pipeline` | `data-pipeline.md`                   |
| #98  | Epic: evaluation pipeline — predict, render, metrics | `evaluation`    | `eval-pipeline.md` (PR #101)         |
| #99  | Epic: R2 integration for datasets and checkpoints    | `storage`       | `eval-pipeline.md` §6                |
| #107 | Epic: training pipeline & ops                        | `training`      | `training-ops-braindump.md` (PR #84) |
| #114 | Epic: codebase modernization                         | `code-health`   | —                                    |
| #148 | Epic: CI & automation platform                       | `ci-automation` | —                                    |
| #149 | Epic: test infrastructure & coverage                 | `testing`       | —                                    |
| #264 | Epic: end-to-end MVP pipeline                        | `training`      | —                                    |
| #321 | Epic: pre-launch cleanup                             | `code-health`   | —                                    |
| #351 | Epic: documentation quality and drift detection      | `documentation` | —                                    |

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

Priority is tracked via a **Priority** single-select field on the project:

| Priority | Definition                    |
| -------- | ----------------------------- |
| P0       | Blocks all progress           |
| P1       | Needed before milestone ships |
| P2       | Planned but not blocking      |
| P3       | Nice-to-have                  |

## 6. Labels

Labels classify issues by **domain**. Type and blocking are handled by native features; priority is a project field.

| Label           | Color   | Description                                                            |
| --------------- | ------- | ---------------------------------------------------------------------- |
| `data-pipeline` | #0e8a16 | Data pipeline work stream                                              |
| `ci-automation` | #1d76db | CI & automation work stream                                            |
| `code-health`   | #fbca04 | Code quality and tech debt                                             |
| `documentation` | #0075ca | Documentation quality, drift detection, and doc-map maintenance        |
| `evaluation`    | #C5DEF5 | Evaluation pipeline, metrics, and inference                            |
| `experiments`   | #E99695 | One-off validation experiments, result replication, and parity studies |
| `monitoring`    | #0E8A16 | Observability, W&B integration, logging                                |
| `storage`       | #D4C5F9 | Storage infrastructure (R2, rclone)                                    |
| `testing`       | #0E8A16 | Test infrastructure, fixtures, CI test config                          |
| `training`      | #8B5CF6 | Training pipeline, ops, and infrastructure                             |

**Multi-label policy:** Most issues carry a single domain label. Cross-cutting infrastructure (e.g., R2/rclone work that serves both data pipeline and eval pipeline) may carry multiple domain labels to appear in all relevant filtered views. Use multiple labels only when the work genuinely spans work streams — not as a default.

Workflow labels (`duplicate`, `invalid`, `wontfix`, `good first issue`, `help wanted`, `question`) are retained for their standard GitHub purposes.

## 7. Milestones

| Milestone            | Work Stream     |
| -------------------- | --------------- |
| data-pipeline v1.0.0 | Data Pipeline   |
| evaluation v1.0.0    | Evaluation      |
| experiments v1.0.0   | Experiments     |
| monitoring v1.0.0    | Monitoring      |
| training v1.0.0      | Training        |
| ci-automation v1.0.0 | CI & Automation |
| code-health v1.0.0   | Code Health     |
| documentation v1.0.0 | Documentation   |
| storage v1.0.0       | Storage         |
| testing v1.0.0       | Testing         |

Every work stream has a milestone. Set the milestone on each issue individually — GitHub does not auto-inherit milestones from parent issues.

### Goal milestones

Goal milestones track measurable quality targets. They coexist with work-stream milestones — use the more specific goal milestone when the issue directly contributes to a measurable target.

| Milestone                | Goal                                  | Phase      |
| ------------------------ | ------------------------------------- | ---------- |
| docstring-coverage-80pct | ≥80% docstring coverage (interrogate) | #114 → #27 |
| zero-bandit-warnings     | Zero bandit security warnings in src/ | #114 → #28 |
| type-annotation-coverage | Type annotations on all public APIs   | #114 → #29 |
| test-coverage-80pct      | Test coverage ≥80% for src/           | #149 → #30 |

## 8. Project

A single org-level GitHub Project contains all issues. Domain labels and saved views replace separate per-domain projects.

### Custom fields

Priority, Start Date, Target Date.

### Built-in fields

Title, Assignees, Status (`Todo` → `In Progress` → `Done` → `Verified`), Labels, Linked PRs, Milestone, Repository, Reviewers, Parent issue, Sub-issues progress.

`Won't Fix` is a terminal status reachable from any prior status (Todo, In Progress, or Done).

`Verified` — done and verified working as expected. The final status for completed work.

`Won't Fix` — issue intentionally closed without resolution (not a bug, out of scope, or superseded by other work). Can be set from any prior status.

### Views

Use saved views with domain label filters to switch between work streams:

| View            | Layout    | Filter                |
| --------------- | --------- | --------------------- |
| All Work        | Hierarchy | (none)                |
| Data Pipeline   | Hierarchy | `label:data-pipeline` |
| Evaluation      | Hierarchy | `label:evaluation`    |
| Experiments     | Hierarchy | `label:experiments`   |
| Storage         | Hierarchy | `label:storage`       |
| CI & Automation | Board     | `label:ci-automation` |
| Code Health     | Table     | `label:code-health`   |
| Documentation   | Table     | `label:documentation` |
| Monitoring      | Table     | `label:monitoring`    |
| Testing         | Table     | `label:testing`       |
| Training        | Hierarchy | `label:training`      |
| Roadmap         | Roadmap   | (none)                |
| Blocked         | Table     | is:blocked            |

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
  ├── member of → PROJECT (single org-level project)
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
8. Link the PR using the appropriate method:

| Method                   | When to use                                   | Auto-closes? |
| ------------------------ | --------------------------------------------- | ------------ |
| `Fixes #N` / `Closes #N` | PR fully resolves the issue                   | Yes          |
| `Refs #N`                | PR is related but doesn't resolve             | No           |
| Development sidebar link | Manual non-closing connection from issue page | No           |

09. When the PR merges, move to **Done** (keep the issue open)
10. After verifying the change works as expected, move to **Verified** and close the issue
11. If the issue is intentionally declined, move to **Won't Fix** and close the issue with a reason comment

## 12. Changes Required

### Org migration

See `docs/org-migration-checklist.md` (PR #116) for the full pre/during/post checklist. Key steps:

- Create GitHub org and transfer repo
- Re-create repo secrets (ANTHROPIC_API_KEY, APPROVAL_BOT_APP_ID, APPROVAL_BOT_PRIVATE_KEY, RUNPOD_API_KEY)

### Post-migration setup

**1. Create a single org-level project** with fields: Priority, Start Date, Target Date. Set up saved views per domain label (see §8).

**2. Add all issues to the project** — add the epic issues; sub-issues appear automatically via hierarchy view.

**3. Set issue types on all existing issues** — assign Epic, Phase, Task, Bug, Feature types via the sidebar or GraphQL:

```bash
# Get issue node ID and type ID, then update
ISSUE_ID=$(gh issue view <num> --json id -q .id)
gh api graphql -f query="
mutation {
  updateIssue(input: {
    id: \"$ISSUE_ID\"
    issueTypeId: \"<type_id>\"
  }) { issue { number issueType { name } } }
}"
```

**4. Migrate blocking relationships to native:**

```bash
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

**5. Set Priority** on all project items from existing label data.

**6. Delete retired labels** (`bug` kept — used by Dependabot):

```bash
gh label delete "enhancement" --yes
gh label delete "blocked" --yes
gh label delete "P0 🔴" --yes
gh label delete "P1 🟠" --yes
gh label delete "P2 🟡" --yes
gh label delete "P3 🔵" --yes
```

**7. Delete old user-level projects** (ktinubu/projects #1–5) after verifying the new org project is set up.
