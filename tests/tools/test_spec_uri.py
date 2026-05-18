"""Tests for the ``synth-setter-spec-uri`` console script.

The script emits the per-job R2 URI for the launcher-uploaded spec — used by
``.github/workflows/generate-dataset-shards.yaml`` to surface that URI as a
workflow output without re-implementing the URI shape in bash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synth_setter.tools import spec_uri


def _valid_spec_payload() -> dict[str, Any]:
    """Return a JSON-serializable dict that validates as DatasetSpec (nested r2).

    :returns: Mapping ready for ``json.dumps`` + ``DatasetSpec.model_validate_json``.
    :rtype: dict[str, Any]
    """
    return {
        "task_name": "test-dataset",
        "run_id": "test-dataset-20260328T120000000Z",
        "created_at": "2026-03-28T12:00:00+00:00",
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [10000, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "intermediate-data",
            "prefix_root": "data",
            "prefix": "data/test-dataset/test-dataset-20260328T120000000Z/",
        },
        "render": {
            "plugin_path": "FakePlugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10000,
        },
    }


def _write_spec(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:  # noqa: DOC101,DOC103
    """Write a spec JSON file under ``tmp_path`` and return the path.

    :returns: Local path to the freshly-written ``input_spec.json``.
    :rtype: Path
    """
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(json.dumps(payload if payload is not None else _valid_spec_payload()))
    return spec_path


class TestComputeSpecUri:
    """``compute_spec_uri`` builds the launcher-uploaded spec URI from a spec path + job name."""

    def test_uri_uses_bucket_from_spec_and_job_name_key(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """Output is ``r2://{bucket}/skypilot-launcher-specs/{job_name}.json``."""
        spec_path = _write_spec(tmp_path)
        uri = spec_uri.compute_spec_uri(spec_path, "smoke-job-1")
        assert uri == "r2://intermediate-data/skypilot-launcher-specs/smoke-job-1.json"

    def test_uri_uses_bucket_from_legacy_flat_keys(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """Pre-PR materialized specs (legacy flat ``r2_bucket``) still resolve correctly."""
        payload = _valid_spec_payload()
        nested = payload.pop("r2")
        payload["r2_bucket"] = nested["bucket"]
        payload["r2_prefix_root"] = nested["prefix_root"]
        payload["r2_prefix"] = nested["prefix"]
        spec_path = _write_spec(tmp_path, payload)
        uri = spec_uri.compute_spec_uri(spec_path, "smoke-job-1")
        assert uri == "r2://intermediate-data/skypilot-launcher-specs/smoke-job-1.json"

    def test_invalid_spec_raises(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """A malformed spec JSON propagates the validation error rather than silently passing."""
        spec_path = tmp_path / "bad.json"
        spec_path.write_text('{"not": "valid"}')
        with pytest.raises(Exception):  # pydantic ValidationError  # noqa: B017
            spec_uri.compute_spec_uri(spec_path, "smoke-job-1")


class TestCli:
    """``main`` is the ``synth-setter-spec-uri`` console-script entrypoint."""

    def test_cli_prints_uri_to_stdout(  # noqa: DOC101,DOC103
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The CLI prints the launcher URI on stdout (one line, no trailing prose)."""
        spec_path = _write_spec(tmp_path)
        rc = spec_uri.main(["--spec", str(spec_path), "--job-name", "smoke-job-1"])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.strip() == (
            "r2://intermediate-data/skypilot-launcher-specs/smoke-job-1.json"
        )

    def test_cli_missing_args_exits_nonzero(self, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: DOC101,DOC103
        """Missing required args surface as a non-zero exit, not a silent empty URI."""
        with pytest.raises(SystemExit) as exc:
            spec_uri.main([])
        assert exc.value.code != 0
