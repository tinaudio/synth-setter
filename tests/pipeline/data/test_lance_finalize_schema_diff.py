"""Pure-function coverage of the shared schema-mismatch rendering.

Runs without the ``fake_r2_remote`` fixture so plugin-less CI (no rclone)
still exercises every diff branch; the e2e drift scenarios live in
``test_lance_finalize.py``.
"""

from __future__ import annotations

import pyarrow as pa

from synth_setter.pipeline.data.lance_shard import schema_mismatch_detail


def test_schema_diff_field_set_drift_names_each_side() -> None:
    """Fields present on only one side are listed under their side."""
    physical = pa.schema([pa.field("wrong", pa.int64())])
    expected = pa.schema([pa.field("audio", pa.float32()), pa.field("mel_spec", pa.float32())])

    detail = schema_mismatch_detail(physical, expected)

    assert "fields only in fragment: ['wrong']" in detail
    assert "fields only in expected: ['audio', 'mel_spec']" in detail


def test_schema_diff_dtype_drift_on_common_field_shows_both_types() -> None:
    """A shared field with different dtypes reports fragment vs expected types."""
    physical = pa.schema([pa.field("audio", pa.float16())])
    expected = pa.schema([pa.field("audio", pa.float32())])

    detail = schema_mismatch_detail(physical, expected)

    assert "field types differ: audio fragment halffloat vs expected float" in detail


def test_schema_diff_metadata_only_drift_appends_skew_hint() -> None:
    """Identical fields with diverging metadata values yield both payloads and the hint."""
    field = pa.field("audio", pa.float32())
    physical = pa.schema([field], metadata={b"k": b"writer-value"})
    expected = pa.schema([field], metadata={b"k": b"validator-value"})

    detail = schema_mismatch_detail(physical, expected)

    assert "metadata 'k': fragment=writer-value expected=validator-value" in detail
    assert "code-version skew" in detail
    assert "rebase onto current main" in detail


def test_schema_diff_metadata_key_missing_on_one_side_renders_absent() -> None:
    """A key present on only one side renders ``<absent>`` for the other."""
    field = pa.field("audio", pa.float32())
    physical = pa.schema([field], metadata={b"k": b"v"})
    expected = pa.schema([field])

    detail = schema_mismatch_detail(physical, expected)

    assert "metadata 'k': fragment=v expected=<absent>" in detail


def test_schema_diff_field_drift_suppresses_skew_hint() -> None:
    """The skew hint stays out of field-level drifts, where corruption is as likely."""
    physical = pa.schema([pa.field("wrong", pa.int64())], metadata={b"k": b"a"})
    expected = pa.schema([pa.field("audio", pa.float32())], metadata={b"k": b"b"})

    detail = schema_mismatch_detail(physical, expected)

    assert "code-version skew" not in detail


def test_schema_diff_nullability_only_drift_reports_field_flags() -> None:
    """A drift invisible to name/type/metadata diffs still yields a non-empty detail."""
    nullable = pa.schema([pa.field("audio", pa.int64(), nullable=True)])
    required = pa.schema([pa.field("audio", pa.int64(), nullable=False)])

    detail = schema_mismatch_detail(nullable, required)

    assert "order or nullability" in detail
    assert "audio" in detail
