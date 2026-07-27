"""Tests for ``OutputFormat`` — the shard-container enum and its extension dispatch."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat


def test_output_format_extension_lance_is_lance() -> None:
    """``LANCE`` shards carry the ``.lance`` suffix."""
    assert OutputFormat.LANCE.extension == ".lance"


def test_from_extension_lance_returns_lance() -> None:
    """``.lance`` reverse-maps to the Lance format."""
    assert OutputFormat.from_extension(".lance") is OutputFormat.LANCE


@pytest.mark.parametrize("suffix", [".parquet", ".h5", ".tar"])
def test_from_extension_unknown_suffix_returns_none(suffix: str) -> None:
    """An unregistered suffix reverse-maps to ``None`` so callers raise their own error.

    Covers the retired HDF5 (``.h5``) and WebDataset (``.tar``) suffixes: now
    that Lance is the only format, they must dispatch to ``None`` like any other
    unknown suffix rather than a live format.

    :param suffix: Filename suffix that must no longer map to a format.
    """
    assert OutputFormat.from_extension(suffix) is None


@pytest.mark.parametrize("fmt", list(OutputFormat))
def test_extension_round_trips_through_from_extension(fmt: OutputFormat) -> None:
    """Every format's ``.extension`` reverse-maps back to that same format.

    :param fmt: The output format under test (swept over every enum member).
    """
    assert OutputFormat.from_extension(fmt.extension) is fmt


def test_value_is_the_lowercase_token() -> None:
    """Enum values are the on-disk / JSON tokens used at the Hydra / R2 boundary."""
    assert OutputFormat.LANCE == "lance"


@pytest.mark.parametrize("member", list(OutputFormat))
def test_member_str_coercion_returns_its_value(member: OutputFormat) -> None:
    """``str(member)`` is the bare value, not ``"OutputFormat.X"`` (``StrEnum`` contract).

    :param member: The output format under test (swept over every enum member).
    """
    assert str(member) == member.value


@pytest.mark.parametrize("legacy_token", ["hdf5", "wds"])
def test_dataset_spec_legacy_output_format_token_rejected(
    legacy_token: str,
    dataset_spec_factory: Callable[..., DatasetSpec],
) -> None:
    """Retired ``output_format`` tokens no longer parse into a ``DatasetSpec``.

    ``hdf5`` and ``wds`` were valid enum values before the format collapse to
    Lance; a spec (from R2 JSON or Hydra) still carrying them must fail loudly.

    :param legacy_token: A retired ``output_format`` string that must be rejected.
    :param dataset_spec_factory: Shared ``conftest`` ``DatasetSpec`` factory.
    """
    with pytest.raises(ValidationError):
        dataset_spec_factory(output_format=legacy_token)
