"""Tests for synth_setter.pipeline.ci.validate_shard's CLI plumbing.

Per-shard Lance validation lives in ``test_validate_shard_lance.py``; this file
covers the not-found guard and the ``main()`` CLI entry point (argv shape and
exit codes), which drives ``validate_all_shards_from_r2``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synth_setter.pipeline.ci import validate_shard as validate_shard_module
from synth_setter.pipeline.ci.validate_shard import validate_all_shards_from_r2, validate_shard
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat


@pytest.fixture()
def real_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    """Build a real Lance DatasetSpec with mocked git/timestamp factories."""
    fixed_now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._get_git_sha", lambda: "a" * 40)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("synth_setter.pipeline.schemas.spec._utc_now", lambda: fixed_now)

    contents = tmp_path / "FakePlugin.vst3" / "Contents"
    contents.mkdir(parents=True)
    (contents / "moduleinfo.json").write_text('{"Version": "1.3.4"}')

    return DatasetSpec(
        task_name="test-dataset",
        output_format=OutputFormat.LANCE,
        train_val_test_sizes=(10, 0, 0),
        base_seed=42,
        r2={"bucket": "intermediate-data"},  # type: ignore[arg-type]
        render={
            "plugin_path": str(tmp_path / "FakePlugin.vst3"),
            "plugin_state_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 44100,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10,
            "gui_toggle_cadence": "never",
        },  # type: ignore[arg-type]
    )


class TestValidateShard:
    """Tests for the ``validate_shard`` dispatcher's guard paths."""

    def test_file_not_found_returns_error(self, real_spec: DatasetSpec, tmp_path: Path) -> None:
        """Path that does not exist returns an error."""
        shard_path = tmp_path / "nonexistent.lance"

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "not found" in errors[0].lower() or "does not exist" in errors[0].lower()

    def test_unsupported_suffix_returns_error(
        self, real_spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A shard whose suffix is not ``.lance`` is rejected naming the registered set.

        :param real_spec: Lance dataset spec fixture.
        :param tmp_path: Per-test tmp dir where the wrong-suffix file is written.
        """
        shard_path = tmp_path / "shard-000000.parquet"
        shard_path.write_bytes(b"payload")

        errors = validate_shard(shard_path, real_spec)

        assert len(errors) == 1
        assert "unsupported shard suffix" in errors[0]

    def test_r2_validation_rejects_unknown_output_format(self, real_spec: DatasetSpec) -> None:
        """An unregistered format cannot silently select a shard validator.

        :param real_spec: Valid spec copied before injecting an unknown format.
        """
        invalid_spec = real_spec.model_copy()
        object.__setattr__(invalid_spec, "output_format", "parquet")

        with pytest.raises(ValueError, match="unsupported output_format"):
            validate_all_shards_from_r2(invalid_spec)


class TestMain:
    """Tests for the CLI entry point ``main()`` with the single-arg shape."""

    def test_cli_rejects_two_args(
        self, real_spec: DatasetSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy 2-arg shape (spec + shard) is rejected."""
        from synth_setter.pipeline.ci.validate_shard import main

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path), "ignored.lance"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_cli_exits_zero_when_all_shards_valid(
        self,
        real_spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valid spec + all-valid shard validation → exit 0.

        :param real_spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        :param monkeypatch: Pytest fixture used to set ``sys.argv`` and stub R2.
        """
        from synth_setter.pipeline.ci.validate_shard import main

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())
        monkeypatch.setattr(validate_shard_module, "validate_all_shards_from_r2", lambda spec: [])

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path)])
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_cli_accepts_file_uri_for_spec_arg(
        self,
        real_spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``file://`` URI for the spec arg loads the same as a bare path → exit 0.

        :param real_spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        :param monkeypatch: Pytest fixture used to set ``sys.argv`` and stub R2.
        """
        from synth_setter.pipeline.ci.validate_shard import main

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())
        monkeypatch.setattr(validate_shard_module, "validate_all_shards_from_r2", lambda spec: [])

        monkeypatch.setattr(sys, "argv", ["validate_shard", spec_json_path.as_uri()])
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_cli_exits_one_when_a_shard_is_invalid(
        self,
        real_spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If any shard in spec.shards fails validation, exit 1.

        :param real_spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        :param monkeypatch: Pytest fixture used to set ``sys.argv`` and stub R2.
        """
        from synth_setter.pipeline.ci.validate_shard import main

        spec_json_path = tmp_path / "spec.json"
        spec_json_path.write_text(real_spec.model_dump_json())
        monkeypatch.setattr(
            validate_shard_module,
            "validate_all_shards_from_r2",
            lambda spec: [f"{spec.shards[0].filename}: path is not a valid Lance dataset"],
        )

        monkeypatch.setattr(sys, "argv", ["validate_shard", str(spec_json_path)])
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
