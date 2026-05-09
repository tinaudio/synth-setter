"""CLI plumbing smoke tests for src/data/vst/generate_vst_dataset.py.

These tests exercise the click surface only — the underlying make_*_dataset
functions are mocked so the tests don't need a real VST plugin or rendering.
Behavioral coverage of the write paths themselves lives in
``test_generate_vst_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from src.data.vst.generate_vst_dataset import main


def _shared_options() -> list[str]:
    """Click options shared by every CLI smoke test below."""
    return [
        "--plugin_path",
        "plugins/Surge XT.vst3",
        "--preset_path",
        "presets/surge-base.vstpreset",
        "--sample_rate",
        "16000",
        "--channels",
        "2",
        "--velocity",
        "100",
        "--signal_duration_seconds",
        "1.0",
        "--min_loudness",
        "-55.0",
        "--param_spec",
        "surge_simple",
        "--sample_batch_size",
        "1",
    ]


def test_h5_extension_routes_positional_to_make_hdf5_dataset(tmp_path: Path) -> None:
    """A ``.h5`` data_file dispatches to ``make_hdf5_dataset`` only."""
    h5_path = tmp_path / "out.h5"
    runner = CliRunner()

    with (
        patch("src.data.vst.generate_vst_dataset.make_hdf5_dataset") as mock_h5,
        patch("src.data.vst.generate_vst_dataset.make_wds_dataset") as mock_wds,
    ):
        result = runner.invoke(
            main,
            [str(h5_path), "1", *_shared_options()],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    mock_h5.assert_called_once()
    mock_wds.assert_not_called()
    assert mock_h5.call_args.kwargs["hdf5_file"] == str(h5_path)


def test_tar_extension_routes_positional_to_make_wds_dataset(tmp_path: Path) -> None:
    """A ``.tar`` data_file dispatches to ``make_wds_dataset`` only."""
    tar_path = tmp_path / "out.tar"
    runner = CliRunner()

    with (
        patch("src.data.vst.generate_vst_dataset.make_hdf5_dataset") as mock_h5,
        patch("src.data.vst.generate_vst_dataset.make_wds_dataset") as mock_wds,
    ):
        result = runner.invoke(
            main,
            [str(tar_path), "1", *_shared_options()],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    mock_wds.assert_called_once()
    mock_h5.assert_not_called()
    assert mock_wds.call_args.kwargs["wds_file"] == str(tar_path)


def test_unknown_extension_is_rejected_with_supported_suffixes_listed(tmp_path: Path) -> None:
    """A data_file with an unsupported extension exits non-zero, naming both supported suffixes."""
    runner = CliRunner()

    result = runner.invoke(
        main,
        [str(tmp_path / "out.parquet"), "1", *_shared_options()],
    )

    assert result.exit_code != 0
    assert "data_file must end in" in result.output
    assert ".h5" in result.output
    assert ".tar" in result.output


def test_neither_writer_called_for_unknown_extension(tmp_path: Path) -> None:
    """Suffix validation fires before either writer is dispatched."""
    runner = CliRunner()

    with (
        patch("src.data.vst.generate_vst_dataset.make_hdf5_dataset") as mock_h5,
        patch("src.data.vst.generate_vst_dataset.make_wds_dataset") as mock_wds,
    ):
        runner.invoke(
            main,
            [str(tmp_path / "out.parquet"), "1", *_shared_options()],
        )

    mock_h5.assert_not_called()
    mock_wds.assert_not_called()


def test_h5_writer_receives_param_spec_object_not_name(tmp_path: Path) -> None:
    """The CLI resolves ``--param_spec`` to a ``ParamSpec`` object before dispatch."""
    h5_path = tmp_path / "out.h5"
    runner = CliRunner()

    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> None:
        captured.update(kwargs)

    with patch("src.data.vst.generate_vst_dataset.make_hdf5_dataset", side_effect=_capture):
        runner.invoke(main, [str(h5_path), "1", *_shared_options()], catch_exceptions=False)

    assert captured["param_spec"].__class__.__name__ == "ParamSpec"
