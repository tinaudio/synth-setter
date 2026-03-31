# Model Promotion Pipeline Reference

> **Status**: NOT IMPLEMENTED — `scripts/promote.py` and
> `.github/workflows/promote.yml` do not exist yet. This document describes
> the planned design. See #122 for tracking.
>
> **Last Updated**: 2026-03-31
> **Tracking**: #122

Promote a W&B run to a GitHub Release. Secrets are documented in [storage-provenance-spec.md](../design/storage-provenance-spec.md) §9.

______________________________________________________________________

## Promote Script (`scripts/promote.py`)

````python
#!/usr/bin/env python3
"""Promote a W&B run to a GitHub Release.

Usage:
    python scripts/promote.py --run-id <wandb_run_id>

Environment variables:
    WANDB_API_KEY     - W&B authentication
    GH_TOKEN          - GitHub authentication for the `gh` CLI
                         (in Actions, set via GH_TOKEN: ${{ secrets.GITHUB_TOKEN }})
    WANDB_ENTITY      - W&B entity (default: tinaudio)
    WANDB_PROJECT     - W&B project (default: synth-setter)
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
        return {"tag": data.get("tagName"), "body": data.get("body")}
    except (json.JSONDecodeError, KeyError):
        return None


def format_release_body(run, config: dict, previous: dict | None) -> str:
    """Format the GitHub Release body with eval card.

    TODO: use `previous` to show metric deltas vs last release.
    """
    metrics = dict(run.summary)
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
    input_artifacts = list(run.used_artifacts())
    for art in input_artifacts:
        body += f"| {art.type} | `{art.name}` (v{art.version}) |\n"

    if not input_artifacts:
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
    model_files = list(Path(tmpdir).rglob("*.pt")) + \
                  list(Path(tmpdir).rglob("*.onnx")) + \
                  list(Path(tmpdir).rglob("*.pth"))
    if model_files:
        return model_files[0]
    all_files = list(Path(tmpdir).rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    return files_only[0] if files_only else None


def promote(run_id: str, entity: str, project: str, repo: str,
            registry_path: str | None = None, dry_run: bool = False):
    """Main promotion logic."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    config = run.config

    tag = get_next_version_tag(repo)
    print(f"Promoting run {run_id} as {tag}")

    if registry_path:
        artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
        if artifacts:
            artifacts[0].link(registry_path, aliases=[tag, "latest"])
            print(f"Linked model to W&B registry: {registry_path}")

    previous = get_previous_release_metrics(repo)
    body = format_release_body(run, config, previous)

    if dry_run:
        print(f"\n--- DRY RUN: Would create release {tag} ---")
        print(body)
        return

    model_path = download_model_artifact(run)

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

    if os.getenv("CI"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"RELEASE_TAG={tag}", file=f)
            print(f"RELEASE_URL={result.stdout.strip()}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a W&B run to GitHub Release")
    parser.add_argument("--run-id", required=True, help="W&B run ID to promote")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", "tinaudio"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "synth-setter"))
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "tinaudio/synth-setter"))
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
````

______________________________________________________________________

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
        uses: actions/checkout@v6

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.10"

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

______________________________________________________________________

## Usage

### From the GitHub UI

Actions tab → "Promote Model" → "Run workflow" → paste run ID → go

### From the CLI

```bash
gh workflow run promote.yml -f run_id=abc123
```

### Dry run (local)

```bash
WANDB_API_KEY=xxx python scripts/promote.py --run-id abc123 --dry-run
```
