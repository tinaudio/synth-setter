"""CLI plumbing smoke tests for src/data/vst/generate_vst_dataset.py.

These tests exercise the click surface only — `make_dataset` is mocked so the
tests don't need a real VST plugin or rendering. Behavioral coverage of the
write path itself lives in `test_generate_vst_dataset.py::test_h5_and_wds_outputs_are_equivalent`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.data.vst.generate_vst_dataset import main


def _shared_options() -> list[str]:
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


def test_h5_extension_calls_make_dataset_with_h5_path_and_no_wds(tmp_path: Path) -> None:
    """A ``.h5`` data_file routes only to ``hdf5_file``."""
    h5_path = tmp_path / "out.h5"
    runner = CliRunner()

    with patch("src.data.vst.generate_vst_dataset.make_dataset") as mock_make:
        result = runner.invoke(
            main,
            [str(h5_path), "1", *_shared_options()],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    mock_make.assert_called_once()
    kwargs = mock_make.call_args.kwargs
    assert kwargs["hdf5_file"] == str(h5_path)
    assert kwargs["wds_file"] is None


def test_tar_extension_routes_positional_to_wds_file(tmp_path: Path) -> None:
    """A ``.tar`` data_file routes to ``wds_file`` and stages h5 in a tmp path."""
    tar_path = tmp_path / "out.tar"
    runner = CliRunner()

    with patch("src.data.vst.generate_vst_dataset.make_dataset") as mock_make:
        result = runner.invoke(
            main,
            [str(tar_path), "1", *_shared_options()],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    mock_make.assert_called_once()
    kwargs = mock_make.call_args.kwargs
    assert kwargs["wds_file"] == tar_path
    assert kwargs["hdf5_file"] is not None
    assert kwargs["hdf5_file"] != str(tar_path)


def test_unknown_extension_is_rejected(tmp_path: Path) -> None:
    """A data_file with an unsupported extension exits non-zero with a clear error."""
    runner = CliRunner()

    result = runner.invoke(
        main,
        [str(tmp_path / "out.parquet"), "1", *_shared_options()],
    )

    assert result.exit_code != 0
    assert ".parquet" in result.output or "data_file" in result.output


@pytest.mark.parametrize("ext", [".h5", ".tar"])
def test_tmp_h5_is_cleaned_up_when_wds(tmp_path: Path, ext: str) -> None:
    """For the wds path, the CLI's internal h5 staging file is unlinked after make_dataset."""
    out = tmp_path / f"out{ext}"
    runner = CliRunner()

    captured: dict[str, object] = {}

    def fake_make_dataset(**kwargs: object) -> None:
        captured["hdf5_file"] = kwargs["hdf5_file"]
        if ext == ".tar":
            Path(str(kwargs["hdf5_file"])).touch()

    with patch(
        "src.data.vst.generate_vst_dataset.make_dataset", side_effect=fake_make_dataset
    ):
        result = runner.invoke(
            main,
            [str(out), "1", *_shared_options()],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    if ext == ".tar":
        h5_tmp = Path(str(captured["hdf5_file"]))
        assert not h5_tmp.exists(), f"wds path leaked tmp h5 file: {h5_tmp}"
