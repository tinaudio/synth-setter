"""Tests for ``synth_setter.pipeline.ci.spec_uri`` — print canonical input_spec R2 URI."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.pipeline.ci.spec_uri import compute_spec_uri, main
from synth_setter.pipeline.schemas.spec import DatasetSpec


def _write_spec(tmp_path: Path, bucket: str = "intermediate-data") -> Path:
    """Materialize a minimal DatasetSpec JSON under ``tmp_path/input_spec.json``.

    :param tmp_path: Directory under which ``input_spec.json`` is written.
    :param bucket: R2 bucket name baked into the spec.
    :return: Path to the written ``input_spec.json``.
    """
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
    """``compute_spec_uri`` returns the spec's canonical under-prefix input_spec URI."""

    def test_returns_canonical_input_spec_uri(self, tmp_path: Path) -> None:
        """Result equals ``spec.r2.input_spec_uri()`` for the loaded spec.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        spec_path = _write_spec(tmp_path)
        spec = DatasetSpec.model_validate_json(spec_path.read_text())
        assert compute_spec_uri(spec_path) == spec.r2.input_spec_uri()

    def test_uri_is_canonical_under_prefix(self, tmp_path: Path) -> None:
        """The returned URI is the canonical ``<prefix>input_spec.json`` form.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        spec_path = _write_spec(tmp_path)
        uri = compute_spec_uri(spec_path)
        assert uri.startswith("r2://intermediate-data/")
        assert uri.endswith("/input_spec.json")

    def test_legacy_flat_spec_still_resolves(self, tmp_path: Path) -> None:
        """Specs already in R2 with flat ``r2_bucket`` keys still parse + emit URI.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
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
            compute_spec_uri(legacy_text)
            == "r2://legacy-bucket/data/t/t-20260328T120000000Z/input_spec.json"
        )


class TestMainCli:
    """CLI entrypoint surface: argv handling, error paths, stdout shape."""

    def test_prints_uri_to_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Happy path: argv resolves spec → exactly one URI line on stdout.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        spec_path = _write_spec(tmp_path)
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(spec_path)])
        main()
        spec = DatasetSpec.model_validate_json(spec_path.read_text())
        out = capsys.readouterr().out.strip()
        assert out == spec.r2.input_spec_uri()

    def test_missing_args_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Wrong-arity invocation surfaces a one-line usage error + exit 1.

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_extra_args_exits_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An extra trailing arg (stale ``cluster_name`` usage) also exits 1.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        spec_path = _write_spec(tmp_path)
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(spec_path), "stale-cluster"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_missing_spec_file_exits_two(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A non-existent spec path exits 2 with a path-named error.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        missing = tmp_path / "missing.json"
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(missing)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert str(missing) in capsys.readouterr().err

    def test_invalid_json_exits_three_without_traceback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Malformed JSON surfaces as exit 3 + one-line stderr (no traceback).

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        bad = tmp_path / "broken.json"
        bad.write_text("{not-json")
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(bad)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 3
        err = capsys.readouterr().err
        assert str(bad) in err
        assert "Traceback" not in err

    def test_schema_violation_exits_three_without_traceback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """JSON that parses but fails DatasetSpec validation also exits 3.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        bad = tmp_path / "invalid.json"
        bad.write_text('{"task_name": "t"}')
        monkeypatch.setattr("sys.argv", ["synth-setter-spec-uri", str(bad)])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 3
        err = capsys.readouterr().err
        assert str(bad) in err
        assert "Traceback" not in err
