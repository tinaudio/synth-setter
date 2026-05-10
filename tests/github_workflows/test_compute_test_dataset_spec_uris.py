"""Pin per-cell ``spec_uri`` derivation in ``test-dataset-generation.yml``.

The Python helper at ``.github/workflows/_compute_test_dataset_spec_uris.py``
maps each (provider, output_format) matrix cell to an ``r2://...`` URI whose
bucket is read from the experiment YAML — defaulting to ``configs/dataset.yaml``
when the experiment doesn't override it. Lock that mapping so a future
refactor can't silently regress to the prior single-default-bucket behavior
that broke validate for bucket-overriding experiments (#883 review round 7).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / ".github" / "workflows" / "_compute_test_dataset_spec_uris.py"


def _run(env: dict[str, str]) -> dict[str, str]:
    """Run the helper and return the parsed ``spec_uris`` map."""
    result = subprocess.run(  # noqa: S603 — fixed argv to a tracked script
        [sys.executable, str(_SCRIPT)],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    line = result.stdout.strip()
    assert line.startswith("spec_uris="), f"unexpected output: {result.stdout!r}"
    return json.loads(line[len("spec_uris=") :])


def _base_env(**overrides: str) -> dict[str, str]:
    env = {
        "EVENT_NAME": "pull_request",
        "DISPATCH_EXPERIMENT": "datagen/runpod-smoke-shard",
        "PROVIDERS": '["skypilot-local"]',
        "OUTPUT_FORMATS": '["hdf5","wds"]',
        "RUN_ID": "42",
        "RUN_ATTEMPT": "1",
        "PATH": "/usr/bin:/bin",
    }
    env.update(overrides)
    return env


class TestSpecUris:
    def test_pr_mode_routes_to_smoke_experiments_default_bucket(self) -> None:
        """PR mode picks runpod-smoke-shard (hdf5) / runpod-smoke-shard-wds (wds); neither
        overrides r2_bucket, so both URIs land in the default bucket."""
        spec_uris = _run(_base_env())
        assert spec_uris == {
            "skypilot-local-hdf5": (
                "r2://intermediate-data/skypilot-launcher-specs/"
                "synth-setter-smoke-skypilot-local-hdf5-42-1.json"
            ),
            "skypilot-local-wds": (
                "r2://intermediate-data/skypilot-launcher-specs/"
                "synth-setter-smoke-skypilot-local-wds-42-1.json"
            ),
        }

    def test_dispatch_mode_with_bucket_overriding_experiment_uses_override(self) -> None:
        """``datagen/10-1k-shards`` overrides r2_bucket → ``experiments``; the helper must thread
        that through instead of falling back to the default."""
        spec_uris = _run(
            _base_env(
                EVENT_NAME="workflow_dispatch",
                DISPATCH_EXPERIMENT="datagen/10-1k-shards",
                PROVIDERS='["runpod","oci"]',
            )
        )
        for key, uri in spec_uris.items():
            assert uri.startswith("r2://experiments/"), f"{key} should use experiments bucket"

    def test_dispatch_mode_with_default_experiment_uses_default_bucket(self) -> None:
        """A dispatch experiment that doesn't override r2_bucket falls back to default."""
        spec_uris = _run(
            _base_env(
                EVENT_NAME="workflow_dispatch",
                DISPATCH_EXPERIMENT="datagen/runpod-smoke-shard",
                PROVIDERS='["runpod"]',
            )
        )
        for uri in spec_uris.values():
            assert uri.startswith("r2://intermediate-data/")

    def test_missing_experiment_yaml_fails_loud(self) -> None:
        """An unknown experiment name raises rather than silently using the default."""
        result = subprocess.run(  # noqa: S603 — fixed argv
            [sys.executable, str(_SCRIPT)],
            env=_base_env(
                EVENT_NAME="workflow_dispatch",
                DISPATCH_EXPERIMENT="datagen/this-experiment-does-not-exist",
                PROVIDERS='["runpod"]',
            ),
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "experiment YAML not found" in result.stderr

    @pytest.mark.parametrize(
        ("output_format", "cluster_format_segment"),
        [("hdf5", "skypilot-local-hdf5"), ("wds", "skypilot-local-wds")],
    )
    def test_pr_mode_emits_per_format_cluster_name(
        self, output_format: str, cluster_format_segment: str
    ) -> None:
        """The output_format threads into the cluster_name segment of the URI so the validate
        matrix's hdf5 cell and wds cell pull distinct spec JSONs."""
        spec_uris = _run(_base_env(OUTPUT_FORMATS=json.dumps([output_format])))
        (uri,) = spec_uris.values()
        assert f"synth-setter-smoke-{cluster_format_segment}-" in uri
