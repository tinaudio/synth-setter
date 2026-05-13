"""Tests for the forward/reverse output-format <-> extension maps in ``spec.py``."""

from __future__ import annotations

from synth_setter.pipeline.schemas.spec import (
    EXTENSION_TO_OUTPUT_FORMAT,
    OUTPUT_FORMAT_TO_EXTENSION,
)


def test_extension_to_output_format_is_inverse_of_forward_map() -> None:
    """``EXTENSION_TO_OUTPUT_FORMAT`` round-trips every entry in the forward map."""
    for output_format, extension in OUTPUT_FORMAT_TO_EXTENSION.items():
        assert EXTENSION_TO_OUTPUT_FORMAT[extension] == output_format


def test_extension_to_output_format_covers_h5_and_tar() -> None:
    """The reverse map dispatches the two formats the pipeline writes today."""
    assert EXTENSION_TO_OUTPUT_FORMAT[".h5"] == "hdf5"
    assert EXTENSION_TO_OUTPUT_FORMAT[".tar"] == "wds"


def test_reverse_map_has_no_silent_collisions() -> None:
    """The reverse map never loses entries to last-key-wins collisions.

    The forward map must use distinct extensions per format, so reversing it
    preserves cardinality. This pins the invariant the import-time guard in
    ``spec.py`` enforces.
    """
    assert len(EXTENSION_TO_OUTPUT_FORMAT) == len(OUTPUT_FORMAT_TO_EXTENSION)
