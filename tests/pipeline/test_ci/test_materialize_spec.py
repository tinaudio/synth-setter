"""Tests for pipeline/ci/materialize_spec.py — composes a DatasetSpec from a Hydra experiment.

Tests are organized around the PUBLIC API:
- main(): parses argv, composes via Hydra, writes input_spec.json under output_dir
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.ci.materialize_spec import main
from pipeline.constants import INPUT_SPEC_FILENAME


class TestMaterializeSpecCli:
    def test_writes_input_spec_json_for_known_experiment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Happy path: a valid experiment name composes into a DatasetSpec written as JSON."""
        monkeypatch.setattr("sys.argv", ["materialize_spec", "ci-materialize-test", str(tmp_path)])
        main()
        out_path = tmp_path / INPUT_SPEC_FILENAME
        assert out_path.is_file()
        spec = json.loads(out_path.read_text())
        assert spec["task_name"] == "ci-materialize-test"

    def test_unknown_experiment_exits_with_concise_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A misspelled experiment surfaces as ``sys.exit(2)`` + a one-line stderr error.

        Mirrors ``skypilot_launch._compose_dataset_spec``'s error-translation contract: CI
        smoke workflows fail with a concise message instead of a multi-page Hydra traceback.
        """
        monkeypatch.setattr(
            "sys.argv",
            ["materialize_spec", "this-experiment-does-not-exist", str(tmp_path)],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "Hydra compose failed for experiment 'this-experiment-does-not-exist'" in err
        assert not (tmp_path / INPUT_SPEC_FILENAME).exists()

    def test_missing_args_exits_with_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Calling with <3 argv elements prints usage to stderr and exits 1."""
        monkeypatch.setattr("sys.argv", ["materialize_spec"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        assert "Usage:" in capsys.readouterr().err
