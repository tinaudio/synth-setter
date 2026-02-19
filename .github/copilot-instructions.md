# Copilot Instructions

## Create GitHub Project and Issues from Branch

The workflow `.github/workflows/create-project-issues.yml` automatically creates a GitHub Project and GitHub Issues based on the word extracted from the current branch name.

### How it works

1. **Trigger**: The workflow runs automatically when a push is made to any `copilot/**` branch, or manually via `workflow_dispatch`.
2. **Word extraction**: The "word" is extracted from the branch name by taking the portion after the last `/`.
   - Example: branch `copilot/create-github-project-issues` → word `create-github-project-issues`
3. **GitHub Project**: A new GitHub Project (Projects v2) is created using the extracted word as the title.
4. **GitHub Issues**: Three issues are created for the extracted word:
   - `Setup: <word>` — Initial setup tasks
   - `Implement: <word>` — Core implementation tasks
   - `Review and Deploy: <word>` — Review and deployment tasks

### Required permissions

- **Issues**: The workflow uses `GITHUB_TOKEN` with `issues: write` permission to create issues.
- **Projects**: Creating a GitHub Project requires a Personal Access Token (PAT) with the `project` scope. Store it as a repository secret named `GH_PROJECT_TOKEN`. If not provided, the workflow will fall back to `GITHUB_TOKEN` and show a warning if project creation fails.

### Manual trigger

You can trigger the workflow manually from the **Actions** tab with an optional custom word:

```
workflow_dispatch → branch_word: <your-word>
```
