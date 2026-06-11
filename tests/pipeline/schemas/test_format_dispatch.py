"""Tests for ``OutputFormat`` — the shard-container enum and its extension dispatch."""

from __future__ import annotations

import pytest

from synth_setter.pipeline.schemas.spec import OutputFormat


def test_output_format_extension_hdf5_is_h5() -> None:
    """``HDF5`` shards carry the ``.h5`` suffix."""
    assert OutputFormat.HDF5.extension == ".h5"


def test_output_format_extension_wds_is_tar() -> None:
    """``WDS`` shards carry the WebDataset ``.tar`` suffix."""
    assert OutputFormat.WDS.extension == ".tar"


def test_output_format_extension_lance_is_lance() -> None:
    """``LANCE`` shards carry the ``.lance`` suffix."""
    assert OutputFormat.LANCE.extension == ".lance"


def test_from_extension_h5_returns_hdf5() -> None:
    """``.h5`` reverse-maps to the HDF5 format."""
    assert OutputFormat.from_extension(".h5") is OutputFormat.HDF5


def test_from_extension_tar_returns_wds() -> None:
    """``.tar`` reverse-maps to the WDS format."""
    assert OutputFormat.from_extension(".tar") is OutputFormat.WDS


def test_from_extension_lance_returns_lance() -> None:
    """``.lance`` reverse-maps to the Lance format."""
    assert OutputFormat.from_extension(".lance") is OutputFormat.LANCE


def test_from_extension_unknown_suffix_returns_none() -> None:
    """An unregistered suffix reverse-maps to ``None`` so callers raise their own error."""
    assert OutputFormat.from_extension(".parquet") is None


@pytest.mark.parametrize("fmt", list(OutputFormat))
def test_extension_round_trips_through_from_extension(fmt: OutputFormat) -> None:
    """Every format's ``.extension`` reverse-maps back to that same format.

    Pins the no-collision invariant the import-time guard in ``spec.py``
    enforces: if two formats shared a suffix, one would fail to round-trip.

    :param fmt: The output format under test (swept over every enum member).
    """
    assert OutputFormat.from_extension(fmt.extension) is fmt


def test_value_is_the_lowercase_token() -> None:
    """Enum values are the on-disk / JSON tokens used at the Hydra / R2 boundary."""
    assert OutputFormat.HDF5 == "hdf5"
    assert OutputFormat.WDS == "wds"
    assert OutputFormat.LANCE == "lance"
