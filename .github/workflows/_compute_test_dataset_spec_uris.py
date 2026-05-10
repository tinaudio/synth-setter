"""Emit a per-matrix-cell ``spec_uris`` map for ``test-dataset-generation.yml``.

The ``setup`` job calls this once. For every (provider, output_format) cell in
the matrix it computes the experiment name (mirroring the inline Hydra-style
expression in ``test-dataset-generation.yml``) and reads ``r2_bucket`` from
that experiment's YAML — falling back to the default in
``configs/dataset.yaml`` when the experiment doesn't override it. The result
is a single JSON object the validate job (and the ``generate-local`` upload
step) look up by ``"<provider>-<output_format>"``.

Without this, the workflow constructed ``spec_uri`` from only the default
bucket. Experiments that override ``r2_bucket`` (e.g.
``datagen/10-1k-shards`` → ``experiments``) would land their specs in bucket
A while validate looked in bucket B, surfacing as a 404 from the validate
job for runpod/oci provider rows.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_YAML = _REPO_ROOT / "configs" / "dataset.yaml"
_EXPERIMENT_DIR = _REPO_ROOT / "configs" / "experiment"


def _experiment_for(output_format: str, event_name: str, dispatch_experiment: str) -> str:
    """Mirror the EXPERIMENT expression in ``test-dataset-generation.yml``."""
    if event_name == "pull_request":
        if output_format == "wds":
            return "datagen/runpod-smoke-shard-wds"
        return "datagen/runpod-smoke-shard"
    return dispatch_experiment


def _bucket_for(experiment: str, default_bucket: str) -> str:
    """Read ``r2_bucket`` from an experiment YAML, defaulting when unset."""
    path = _EXPERIMENT_DIR / f"{experiment}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"experiment YAML not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected mapping at top level, got {type(data).__name__}")
    bucket = data.get("r2_bucket", default_bucket)
    if not isinstance(bucket, str) or not bucket.strip():
        raise ValueError(f"{path}: r2_bucket must be a non-blank string, got {bucket!r}")
    return bucket


def main() -> int:
    providers = json.loads(os.environ["PROVIDERS"])
    output_formats = json.loads(os.environ["OUTPUT_FORMATS"])
    event_name = os.environ["EVENT_NAME"]
    dispatch_experiment = os.environ["DISPATCH_EXPERIMENT"]
    run_id = os.environ["RUN_ID"]
    run_attempt = os.environ["RUN_ATTEMPT"]

    with _DATASET_YAML.open() as f:
        default_bucket = yaml.safe_load(f)["r2_bucket"]

    spec_uris: dict[str, str] = {}
    for provider in providers:
        for output_format in output_formats:
            cluster = f"synth-setter-smoke-{provider}-{output_format}-{run_id}-{run_attempt}"
            experiment = _experiment_for(output_format, event_name, dispatch_experiment)
            bucket = _bucket_for(experiment, default_bucket)
            spec_uris[f"{provider}-{output_format}"] = (
                f"r2://{bucket}/skypilot-launcher-specs/{cluster}.json"
            )

    sys.stdout.write(f"spec_uris={json.dumps(spec_uris, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
