---
name: wave-orchestration
description: Orchestrates batches of independent PRs through dependency-aware waves with parallel sub-agents. Use this skill when you have multiple improvements, features, or fixes to implement as separate PRs and need to manage their dependencies, parallelize work across sub-agents, and verify each PR. Trigger when the user asks to "batch PRs", "launch waves", "parallelize PRs", "implement multiple changes", or when you have 3+ independent tasks that each need their own PR. Also trigger on "dependency graph", "wave 1/2/3", or "parallel agents for PRs".
---

# Wave Orchestration Skill

This skill orchestrates multiple PRs through dependency-aware waves with parallel sub-agents and structured verification.

## When to use

You have N improvements/fixes/features. Each needs its own PR. Some depend on others. You want to:
1. Create tracking issues for all of them
2. Map the dependency graph
3. Dispatch independent PRs in parallel waves
4. Verify each PR before the next wave
5. Gate waves on merge

## Process

### Phase 1: Inventory and dependency graph

1. List all items with their dependencies
2. For each item, list the files it will create or modify
3. Build a dependency graph:
   ```
   Independent (no blockers):  → Wave 1
   Depends on Wave 1 items:    → Wave 2
   Depends on Wave 2 items:    → Wave 3
   ```
4. **File-overlap check**: Before finalizing waves, check that no two PRs in the same wave touch the same file. If they do, bump one to the next wave and note the reason:
   ```
   CONFLICT: PR-A and PR-B both modify pyproject.toml
   → Move PR-B to Wave 2 (PR-A has higher priority / fewer dependencies)
   ```
   This reduces the risk of merge conflicts when parallel PR branches are merged (the worktrees are just local checkouts used to implement those branches). Check all file types — config files like `pyproject.toml`, `Makefile`, `conftest.py`, and workflow YAMLs are the most common sources of overlap.
5. Identify the maximum parallelism per wave (recommend 6 agents max)

### Phase 2: Issue creation

For each item:
1. Create a GitHub issue (invoke `/github-taxonomy` for compliance)
2. Set issue type, domain label, milestone, project board, priority
3. Set blocking relationships between dependent issues
4. All issues created before any PRs — gives a complete view of planned work

### Phase 3: Wave execution

For each wave:
1. **Pre-flight**: Verify all blocking PRs from previous wave are merged. Pull latest main.
2. **Dispatch**: Launch N parallel sub-agents, each in an isolated git worktree created via `git worktree add --detach ../worktrees/<wave>-<task> origin/main` (e.g., `../worktrees/wave1-ci-add-codeowners`).
3. Each sub-agent:
   - Creates a human-readable branch (e.g., `ci/add-codeowners`, `test/add-pytest-xdist`)
   - Makes the change
   - Commits (no Co-Authored-By, no "Generated with Claude Code" footer)
   - Pushes and creates a PR linking to its tracking issue (`Closes #N`)
   - PR body includes: rationale, pros/cons, verification steps
4. **Verify**: After all agents complete, invoke `/pr-checkbox` to verify each PR behaviorally (run tools, not grep diffs)
5. **Report**: Present results to user with pass/fail summary
6. **Gate**: Wait for user to review and merge before proceeding to next wave

### Phase 4: Review comment handling

After PR creation:
1. Check all PRs for review comments (Copilot, maintainer, bot)
2. Analyze each comment: agree, disagree, or partially agree
3. Dispatch fix agents for PRs needing changes
4. Reply to all inline comments with resolution
5. Re-verify fixed PRs

### Sub-agent prompt template

Each sub-agent needs:
- Issue number and branch name
- Full task description with exact changes
- File paths to read first
- Commit message format
- PR body requirements (rationale, pros/cons, `Closes #N`)
- Label and milestone
- Constraints (no Co-Authored-By, no Claude footer)

### Branch naming convention

Use human-readable names that indicate the domain and purpose:
- `ci/add-codeowners` — CI infrastructure
- `test/add-pytest-xdist` — Testing infrastructure
- `build/optional-dependency-groups` — Build system
- `dev/add-devcontainer` — Developer experience

### Verification

After each wave, invoke `/pr-checkbox` which enforces:
- Behavioral checks (run the tool, not grep the diff)
- Escalation hierarchy (Level 1 first, descend with justification)
- Checkboxes with actual commands and console output
- Unchecked boxes for ambiguous results

### Cleanup

Between waves:
- Clean up completed worktrees: `git worktree remove <path>`
- Delete merged local branches: `git branch -D <branch>`
- Prune stale worktree references: `git worktree prune`

## Anti-patterns

- Batching multiple unrelated changes/items into one PR — keep each independent change in its own PR
- Starting Wave N+1 before Wave N is merged — gate on merge
- Dispatching agents without tracking issues — issues first, PRs second
- Skipping verification — every PR gets pr-checkbox verification
- Using `main` worktree for sub-agent work — always isolate in worktrees
- Putting PRs that touch the same file in the same wave — file overlap causes merge conflicts; bump one to the next wave
