"""Tests for ``synth_setter.pipeline.ci.spec_uri`` — print canonical input_spec R2 URI."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.pipeline.ci.spec_uri import (
    compute_spec_uri,
    compute_spec_uri_from_hydra,
    main,
)
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
            "sample_rate": 44100,
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
            '"renderer_version":"v","sample_rate":44100,"channels":1,"velocity":1,'
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


_HYDRA_EXPERIMENT = "generate_dataset/smoke-shard"


class TestComputeSpecUriFromHydra:
    """``compute_spec_uri_from_hydra`` derives the URI by Hydra-composing the dataset cfg."""

    def test_matches_launcher_side_computation(self) -> None:
        """URI equals what ``synth-setter-generate-dataset`` would compute via ``spec_from_cfg``.

        Pins the load-bearing claim of this whole feature: the helper and the
        launcher exercise the same ``DatasetSpec.from_hydra_cfg`` derivation, so
        a CI cell consuming this URI will hit the same R2 object the launcher
        writes. Catches any future divergence in cfg-to-spec construction.
        """
        from hydra import compose, initialize_config_module

        from synth_setter.cli.generate_dataset import spec_from_cfg

        overrides = [f"experiment={_HYDRA_EXPERIMENT}", "+run_id=parity-test"]
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            cfg = compose(config_name="dataset", overrides=overrides)

        launcher_uri = spec_from_cfg(cfg).r2.input_spec_uri()
        helper_uri = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "parity-test")
        assert helper_uri == launcher_uri

    def test_run_id_override_lands_in_uri(self) -> None:
        """The ``run_id_override`` argument appears verbatim in the under-prefix URI."""
        uri = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "cell-runpod-hdf5")
        assert "/cell-runpod-hdf5/input_spec.json" in uri
        assert uri.endswith("/input_spec.json")
        assert uri.startswith("r2://")

    def test_uri_invariant_to_created_at_when_run_id_pinned(self) -> None:
        """Two calls with the same ``run_id_override`` produce identical URIs.

        The default ``created_at`` factory fires fresh on each compose, so the
        only way both calls produce equal URIs is if ``run_id`` (pinned)
        suppresses the ``created_at``-derived run_id factory and the prefix
        derivation ignores ``created_at``. This pins the invariant the
        downstream "validator computes URI from matrix coords" flow relies on.
        """
        uri_a = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "pinned")
        uri_b = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "pinned")
        assert uri_a == uri_b

    def test_distinct_run_ids_produce_distinct_uris(self) -> None:
        """Different ``run_id_override`` values produce different URIs."""
        uri_a = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "cell-a")
        uri_b = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "cell-b")
        assert uri_a != uri_b
        assert "/cell-a/" in uri_a and "/cell-b/" in uri_b

    def test_uri_format_matches_canonical_prefix_template(self) -> None:
        """URI follows ``r2://<bucket>/data/<task>/<run>/input_spec.json`` exactly."""
        uri = compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "abc12345")
        assert uri.endswith("/data/smoke-shard/abc12345/input_spec.json")


class TestCliFromExperimentMode:
    """``--from-experiment EXP --run-id-override RUNID`` CLI mode."""

    def test_prints_uri_to_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Happy path: both flags set → exactly one URI line on stdout.

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-spec-uri",
                "--from-experiment",
                _HYDRA_EXPERIMENT,
                "--run-id-override",
                "cli-cell-42",
            ],
        )
        main()
        out = capsys.readouterr().out.strip()
        assert out == compute_spec_uri_from_hydra(_HYDRA_EXPERIMENT, "cli-cell-42")

    def test_equals_form_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--flag=value`` form is honored alongside the space-separated form.

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-spec-uri",
                f"--from-experiment={_HYDRA_EXPERIMENT}",
                "--run-id-override=eq-form-cell",
            ],
        )
        main()
        out = capsys.readouterr().out.strip()
        assert "/eq-form-cell/" in out

    def test_from_experiment_requires_run_id_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--from-experiment`` alone surfaces a usage error + exit 1.

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr(
            "sys.argv",
            ["synth-setter-spec-uri", "--from-experiment", _HYDRA_EXPERIMENT],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_run_id_override_requires_from_experiment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--run-id-override`` alone surfaces a usage error + exit 1.

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr(
            "sys.argv",
            ["synth-setter-spec-uri", "--run-id-override", "orphaned"],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_positional_and_flag_modes_are_mutually_exclusive(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Mixing positional ``spec.json`` with ``--from-experiment`` exits 1.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        spec_path = _write_spec(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-spec-uri",
                str(spec_path),
                "--from-experiment",
                _HYDRA_EXPERIMENT,
                "--run-id-override",
                "mixed",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert "Usage:" in capsys.readouterr().err

    def test_unknown_experiment_exits_three_without_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Hydra failures (missing experiment, bad override) collapse to exit 3.

        Reuses ``_EXIT_INVALID_SPEC`` so log scanners that already grep exit 3
        for "spec didn't validate" pick up compose failures too — both signal
        "the caller couldn't get a usable spec from the input".

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        import uuid

        # Randomized name so a future contributor naming a real experiment can't
        # silently flip this test from negative-case to positive-case.
        bogus = f"generate_dataset/does-not-exist-{uuid.uuid4().hex}"
        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-spec-uri",
                "--from-experiment",
                bogus,
                "--run-id-override",
                "x",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 3
        err = capsys.readouterr().err
        assert "Traceback" not in err

    def test_experiment_named_usage_does_not_trip_sentinel(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--from-experiment __usage__`` reaches Hydra, not the sentinel branch.

        Defends the upgrade from the prior ``"__usage__"`` magic string to a
        class-based sentinel: a user passing ``--from-experiment __usage__``
        now hits Hydra compose (which fails because no such experiment file
        exists, exit 3) rather than being misrouted into the usage-error
        branch (exit 1).

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr(
            "sys.argv",
            [
                "synth-setter-spec-uri",
                "--from-experiment",
                "__usage__",
                "--run-id-override",
                "x",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 3, "should reach compose, not the usage branch"
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "__usage__" in err

    def test_usage_text_uses_live_argv_program_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Usage banner uses ``Path(sys.argv[0]).name`` rather than hard-coding the script name.

        A ``python -m synth_setter.pipeline.ci.spec_uri`` invocation (or any
        non-default entrypoint) should display the actual command the operator
        just ran.

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        monkeypatch.setattr("sys.argv", ["/usr/local/bin/aliased-spec-uri"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "aliased-spec-uri" in err
        assert "synth-setter-spec-uri" not in err
