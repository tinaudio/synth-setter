# Model Promotion Pipeline Reference

Everything you need to implement a promote-and-release workflow for `synth-permutations`.
Results live in W&B during development; GitHub Releases become the permanent record when you promote.

---

## Architecture Overview

```
Day-to-day development:
  train.py → logs metrics + model artifact to W&B
  eval.py  → logs eval metrics to W&B

When a model is good enough:
  You trigger: gh workflow run promote.yml -f run_id=abc123
    →
    ├─ 1. Pull run metrics + config from W&B API
    ├─ 2. Download model artifact from W&B
    ├─ 3. (Optional) Link artifact to W&B model registry
    ├─ 4. Create GitHub Release with:
    │      - Tag: model-v{N}
    │      - Body: eval card (metrics, dataset, config, W&B link)
    │      - Asset: model file (e.g. model.onnx, model.pt)
    └─ 5. Update README badge or one-liner with current model version
```

---

## Training Script Requirements

Your training script needs to log two things for the pipeline to work:

```python
import wandb, os

run = wandb.init(
    project="synth-permutations",
    config={
        # your hyperparams
        "lr": 3e-4,
        "epochs": 100,
        # git traceability
        "github_sha": os.environ.get("GITHUB_SHA", "local"),
    }
)

# ... train ...

# Log final metrics to summary (these get pulled at promote time)
wandb.summary["mse"] = final_mse
wandb.summary["spectral_convergence"] = final_sc
wandb.summary["param_accuracy"] = final_acc

# Log the model as an artifact
artifact = wandb.Artifact(f"model-{run.id}", type="model")
artifact.add_file("model.pt")
run.log_artifact(artifact)

run.finish()
```

The key contract: whatever keys you put in `wandb.summary`, those are what
the promote script will pull and put in the release body.

---

## Promote Script (`scripts/promote.py`)

This is the core script. It takes a W&B run ID, pulls everything it needs,
and creates a GitHub Release.

```python
#!/usr/bin/env python3
"""Promote a W&B run to a GitHub Release.

Usage:
    python scripts/promote.py --run-id <wandb_run_id>

Environment variables:
    WANDB_API_KEY     - W&B authentication
    GITHUB_TOKEN      - GitHub authentication (provided automatically in Actions)
    WANDB_ENTITY      - W&B entity (default: your username)
    WANDB_PROJECT     - W&B project (default: synth-permutations)
"""

import argparse, json, os, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path

import wandb


def get_next_version_tag(repo: str) -> str:
    """Determine next model-vN tag by listing existing releases."""
    result = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--limit", "100"],
        capture_output=True, text=True
    )
    existing = [
        line.split("\t")[2]  # tag is third column
        for line in result.stdout.strip().split("\n")
        if line and "model-v" in line
    ]
    if not existing:
        return "model-v1"
    versions = [int(t.replace("model-v", "")) for t in existing]
    return f"model-v{max(versions) + 1}"


def get_previous_release_metrics(repo: str) -> dict | None:
    """Pull metrics from the most recent release body for delta comparison."""
    result = subprocess.run(
        ["gh", "release", "view", "--repo", repo, "--json", "body,tagName"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        # Parse the metrics from the release body
        # (This depends on your release body format - keep it parseable)
        return {"tag": data.get("tagName"), "body": data.get("body")}
    except (json.JSONDecodeError, KeyError):
        return None


def format_release_body(run, config: dict, previous: dict | None) -> str:
    """Format the GitHub Release body with eval card."""

    # Pull all summary metrics
    metrics = dict(run.summary)
    # Remove wandb internal keys
    metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}

    body = f"""## Eval Card

| Field | Value |
|-------|-------|
| **W&B Run** | [{run.id}]({run.url}) |
| **Date** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |
| **Git SHA** | `{config.get('github_sha', 'unknown')}` |

### Metrics

| Metric | Value |
|--------|-------|
"""
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            body += f"| {k} | {v:.6f} |\n"
        else:
            body += f"| {k} | {v} |\n"

    body += f"""
### Config

```json
{json.dumps(dict(config), indent=2, default=str)}
```

### Dataset

| Field | Value |
|-------|-------|
"""
    # Pull dataset artifact info if logged
    input_artifacts = run.used_artifacts()
    for art in input_artifacts:
        body += f"| {art.type} | `{art.name}` (v{art.version}) |\n"

    if not list(input_artifacts):
        body += "| (none logged) | — |\n"

    return body


def download_model_artifact(run) -> Path | None:
    """Download the model artifact from the run, return local path."""
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        print("WARNING: No model artifact found on this run")
        return None
    if len(artifacts) > 1:
        print(f"WARNING: {len(artifacts)} model artifacts found, using first")

    art = artifacts[0]
    tmpdir = tempfile.mkdtemp()
    art.download(root=tmpdir)
    # Find the model file
    model_files = list(Path(tmpdir).rglob("*.pt")) + \
                  list(Path(tmpdir).rglob("*.onnx")) + \
                  list(Path(tmpdir).rglob("*.pth"))
    if model_files:
        return model_files[0]
    # If no known extension, just return the directory
    all_files = list(Path(tmpdir).rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    return files_only[0] if files_only else None


def promote(run_id: str, entity: str, project: str, repo: str,
            registry_path: str | None = None, dry_run: bool = False):
    """Main promotion logic."""

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    config = run.config

    # 1. Determine version
    tag = get_next_version_tag(repo)
    print(f"Promoting run {run_id} as {tag}")

    # 2. Optional: link to W&B model registry
    if registry_path:
        artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
        if artifacts:
            artifacts[0].link(registry_path, aliases=[tag, "latest"])
            print(f"Linked model to W&B registry: {registry_path}")

    # 3. Format release body
    previous = get_previous_release_metrics(repo)
    body = format_release_body(run, config, previous)

    if dry_run:
        print(f"\n--- DRY RUN: Would create release {tag} ---")
        print(body)
        return

    # 4. Download model artifact
    model_path = download_model_artifact(run)

    # 5. Create GitHub Release
    cmd = [
        "gh", "release", "create", tag,
        "--repo", repo,
        "--title", f"{tag}: {run.name}",
        "--notes", body,
    ]
    if model_path:
        cmd.append(str(model_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR creating release: {result.stderr}")
        raise SystemExit(1)

    print(f"Release created: {result.stdout.strip()}")

    # 6. Write outputs for GitHub Actions
    if os.getenv("CI"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"RELEASE_TAG={tag}", file=f)
            print(f"RELEASE_URL={result.stdout.strip()}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a W&B run to GitHub Release")
    parser.add_argument("--run-id", required=True, help="W&B run ID to promote")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "your-entity"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "synth-permutations"))
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "your-user/synth-permutations"))
    parser.add_argument("--registry", default=None, help="Optional W&B registry path")
    parser.add_argument("--dry-run", action="store_true", help="Print release body without creating")
    args = parser.parse_args()

    promote(
        run_id=args.run_id,
        entity=args.entity,
        project=args.project,
        repo=args.repo,
        registry_path=args.registry,
        dry_run=args.dry_run,
    )
```

---

## GitHub Actions Workflow (`.github/workflows/promote.yml`)

```yaml
name: Promote Model

on:
  workflow_dispatch:
    inputs:
      run_id:
        description: "W&B run ID to promote"
        required: true
        type: string
      registry:
        description: "Optional W&B registry path (leave empty to skip)"
        required: false
        type: string
      dry_run:
        description: "Dry run (print release body without creating)"
        required: false
        type: boolean
        default: false

permissions:
  contents: write  # needed to create releases

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install wandb

      - name: Promote model
        id: promote
        run: |
          python scripts/promote.py \
            --run-id "${{ inputs.run_id }}" \
            --repo "${{ github.repository }}" \
            ${{ inputs.registry && format('--registry "{0}"', inputs.registry) || '' }} \
            ${{ inputs.dry_run && '--dry-run' || '' }}
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Summary
        if: ${{ !inputs.dry_run }}
        run: |
          echo "## Model Promoted" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "- **Tag**: ${{ steps.promote.outputs.RELEASE_TAG }}" >> $GITHUB_STEP_SUMMARY
          echo "- **Release**: ${{ steps.promote.outputs.RELEASE_URL }}" >> $GITHUB_STEP_SUMMARY
          echo "- **W&B Run**: ${{ inputs.run_id }}" >> $GITHUB_STEP_SUMMARY
```

---

## Usage

### From the GitHub UI

Actions tab → "Promote Model" → "Run workflow" → paste run ID → go

### From the CLI

```bash
gh workflow run promote.yml -f run_id=abc123
```

### Dry run (local, no GitHub release created)

```bash
WANDB_API_KEY=xxx python scripts/promote.py --run-id abc123 --dry-run
```

---

## What Gets Created

A GitHub Release at `github.com/you/synth-permutations/releases/tag/model-v3` with:

- **Title**: `model-v3: run-name-from-wandb`
- **Tag**: `model-v3` (on the current default branch HEAD)
- **Body**: Eval card with metrics table, config dump, dataset versions, W&B link
- **Asset**: The model file (model.pt / model.onnx) downloadable via the release

---

## Downstream: CLI Model Download

Your future CLI can pull the latest model without any W&B dependency:

```python
import subprocess, json

result = subprocess.run(
    ["gh", "release", "view", "--repo", "you/synth-permutations",
     "--json", "assets,tagName"],
    capture_output=True, text=True
)
release = json.loads(result.stdout)
# download the model asset from release["assets"][0]["url"]
```

Or for end users who don't have `gh`:

```
https://github.com/you/synth-permutations/releases/latest/download/model.pt
```

This URL always resolves to the latest release's model file. No auth needed
for public repos.

---

## Secrets Required

| Secret | Where to get it |
|--------|-----------------|
| `WANDB_API_KEY` | wandb.ai/settings → API Keys |
| `GITHUB_TOKEN` | Provided automatically by GitHub Actions |

---

## What You Don't Need

- `compare_runs.py` — solves a team review coordination problem you don't have.
  You're already looking at W&B when deciding to promote.
- `deploy.yaml` / `deployment.py` — GitHub Deployment objects are metadata for
  multi-environment staging→production flows. Overkill for solo work.
- ChatOps (`/promote` slash commands in PR comments) — `workflow_dispatch` with
  the GitHub UI or `gh` CLI is simpler and less magic.
- CML — useful if you want auto-generated PR comments with plots on every push.
  Not needed for the promote flow.
- `$GITHUB_STEP_SUMMARY` for eval results — the release body IS the permanent
  record. The step summary is just a confirmation that the workflow ran.
