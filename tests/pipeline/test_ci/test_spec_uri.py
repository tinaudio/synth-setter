"""Tests for ``synth_setter.pipeline.ci.spec_uri`` — print launcher's spec R2 URI."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.pipeline.ci.spec_uri import compute_spec_uri, main
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _write_spec(  # noqa: DOC101,DOC103,DOC201,DOC203
    tmp_path: Path, bucket: str = "intermediate-data"
) -> Path:
    """Materialize a minimal DatasetSpec JSON under ``tmp_path/input_spec.json``."""
    spec = DatasetSpec(
        task_name="ci-task",
        output_format="hdf5",
        train_val_test_sizes=(1, 0, 0),
        base_seed=42,
        r2={"bucket": bucket},  # type: ignore[arg-type]
        render={  # type: ignore[arg-type]
            "plugin_path": str(tmp_path / "plugin.vst3"),
            "preset_path": str(tmp_path / "preset.vstpreset"),
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 1,
            "velocity": 64,
            "signal_duration_seconds": 1.0,
            "min_loudness": -30.0,
            "samples_per_render_batch": 1,
            "samples_per_shard": 1,
        },
    )
    spec_path = tmp_path / "input_spec.json"
    spec_path.write_text(spec.model_dump_json())
    return spec_path


class TestComputeSpecUri:
    """``compute_spec_uri`` builds the launcher's per-job R2 URI from spec + cluster."""

    def test_returns_canonical_launcher_uri(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """URI matches ``r2://<bucket>/skypilot-launcher-specs/<cluster>.json`` exactly."""
        spec_path = _write_spec(tmp_path)
        assert (
            compute_spec_uri(spec_path, "my-cluster-7")
            == "r2://intermediate-data/skypilot-launcher-specs/my-cluster-7.json"
        )

    def test_legacy_flat_spec_still_resolves(self, tmp_path: Path) -> None:  # noqa: DOC101,DOC103
        """Specs already in R2 with flat ``r2_bucket`` keys still parse + emit URI."""
        legacy_text = tmp_path / "legacy.json"
        legacy_text.write_text(
            '{"task_name":"t","run_id":"t-20260328T120000000Z",'
            '"created_at":"2026-03-28T12:00:00+00:00",'
            '"git_sha":"a000000000000000000000000000000000000000","is_repo_dirty":false,'
            '"output_format":"hdf5","train_val_test_sizes":[1,0,0],'
            '"train_val_test_seeds":null,"base_seed":42,'
            '"r2_bucket":"legacy-bucket","r2_prefix_root":"data",'
            '"r2_prefix":"data/t/t-20260328T120000000Z/",'
            '"render":{"plugin_path":"x","preset_path":"x","param_spec_name":"surge_simple",'
            '"renderer_version":"v","sample_rate":16000,"channels":1,"velocity":1,'
            '"signal_duration_seconds":1.0,"min_loudness":-1.0,"samples_per_render_batch":1,'
            '"samples_per_shard":1}}'
        )
        assert (
            compute_spec_uri(legacy_text, "cluster-x")
            == "r2://legacy-bucket/skypilot-launcher-specs/cluster-x.json"
        )


class TestMainCli:
    """CLI entrypoint surface: argv handling, error paths, stdout shape."""

    def test_prints_uri_to_stdout(  # noqa: DOC101,DOC103
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Happy path: argv resolves spec + cluster → exactly one URI line on stdout."""
        spec_path = _write_spec(tmp_path)
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(spec_path), "cluster-z"])
        main()
        out = capsys.readouterr().out.strip()
        assert out == "r2://intermediate-data/skypilot-launcher-specs/cluster-z.json"

    def test_missing_args_exits_one(  # noqa: DOC101,DOC103
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Wrong-arity invocation surfaces a one-line usage error + exit 1."""
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_missing_spec_file_exits_two(  # noqa: DOC101,DOC103
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A non-existent spec path exits 2 with a path-named error."""
        missing = tmp_path / "missing.json"
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(missing), "cluster-a"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert str(missing) in capsys.readouterr().err
