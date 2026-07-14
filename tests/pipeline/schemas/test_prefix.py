"""Tests for ``synth_setter.pipeline.schemas.prefix``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from synth_setter.pipeline.schemas.prefix import (
    DatasetConfigId,
    DatasetRunId,
    assert_r2_prefix_matches,
    make_dataset_wandb_run_id,
    make_r2_prefix,
)

FIXED_NOW = datetime(2026, 6, 15, 12, 30, 45, tzinfo=UTC)


class TestMakeDatasetWandbRunId:
    """Tests for make_dataset_wandb_run_id."""

    def test_make_run_id_fixed_timestamp_format(self):
        """Fixed UTC timestamp (microsecond=0) produces the expected run ID string."""
        ts = datetime(2026, 3, 13, 10, 0, 0, tzinfo=UTC)
        result = make_dataset_wandb_run_id(DatasetConfigId("surge-simple-480k-10k"), timestamp=ts)
        assert result == "surge-simple-480k-10k-20260313T100000000Z"

    def test_make_run_id_includes_millisecond_field(self):
        """A timestamp with microseconds produces a 3-digit millisecond suffix."""
        ts = datetime(2026, 5, 3, 13, 36, 33, 456789, tzinfo=UTC)
        result = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts)
        assert result == "cfg-20260503T133633456Z"

    def test_make_run_id_milliseconds_disambiguate_within_same_second(self):
        """Two timestamps within the same wall-clock second produce different IDs."""
        ts1 = datetime(2026, 3, 13, 10, 0, 0, 100_000, tzinfo=UTC)
        ts2 = datetime(2026, 3, 13, 10, 0, 0, 900_000, tzinfo=UTC)
        id1 = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts1)
        id2 = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts2)
        assert id1 != id2

    @patch("synth_setter.pipeline.schemas.prefix._utc_now", return_value=FIXED_NOW)
    def test_make_run_id_default_timestamp_is_utc(self, mock_utc_now):
        """Default (no timestamp) stamps the current UTC time and produces the expected ID.

        :param mock_utc_now: Patched ``prefix._utc_now`` returning ``FIXED_NOW``.
        """
        result = make_dataset_wandb_run_id(DatasetConfigId("test-config"))
        mock_utc_now.assert_called_once_with()
        assert result == "test-config-20260615T123045000Z"

    def test_make_run_id_rejects_naive_timestamp(self):
        """Naive datetime (no tzinfo) raises ValueError."""

        naive = datetime(2026, 3, 13, 10, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=naive)

    def test_make_run_id_rejects_non_utc_timezone(self):
        """Non-UTC timezone raises ValueError."""

        non_utc = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone(timedelta(hours=5)))
        with pytest.raises(ValueError, match="must be UTC"):
            make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=non_utc)

    def test_make_run_id_deterministic_same_inputs(self):
        """Same arguments produce the same result."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        a = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts)
        b = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts)
        assert a == b

    def test_make_run_id_seconds_precision(self):
        """Timestamps one second apart produce different IDs."""
        ts1 = datetime(2026, 3, 13, 10, 0, 0, tzinfo=UTC)
        ts2 = datetime(2026, 3, 13, 10, 0, 1, tzinfo=UTC)
        id1 = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts1)
        id2 = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts2)
        assert id1 != id2


class TestMakeR2Prefix:
    """Tests for make_r2_prefix."""

    def test_make_r2_prefix_format(self):
        """Prefix matches the expected data/<config_id>/<run_id>/ pattern."""
        result = make_r2_prefix(
            DatasetConfigId("surge-simple-480k-10k"),
            DatasetRunId("surge-simple-480k-10k-20260313T100000000Z"),
        )
        assert result == "data/surge-simple-480k-10k/surge-simple-480k-10k-20260313T100000000Z/"

    def test_make_r2_prefix_trailing_slash(self):
        """Prefix always ends with a trailing slash."""
        result = make_r2_prefix(DatasetConfigId("a"), DatasetRunId("b"))
        assert result.endswith("/")

    def test_make_r2_prefix_strips_trailing_slash_from_root(self):
        """``prefix_root="data/"`` does not produce a double slash."""
        result = make_r2_prefix(DatasetConfigId("a"), DatasetRunId("b"), prefix_root="data/")
        assert result == "data/a/b/"

    def test_make_r2_prefix_strips_leading_slash_from_root(self):
        """``prefix_root="/data"`` does not produce a leading slash."""
        result = make_r2_prefix(DatasetConfigId("a"), DatasetRunId("b"), prefix_root="/data")
        assert result == "data/a/b/"

    def test_make_r2_prefix_rejects_empty_root(self):
        """Slash-only or empty ``prefix_root`` raises rather than producing ``/a/b/``."""

        for bad in ("", "/", "///"):
            with pytest.raises(ValueError, match="prefix_root"):
                make_r2_prefix(DatasetConfigId("a"), DatasetRunId("b"), prefix_root=bad)


_ASSERT_CONFIG_ID = DatasetConfigId("surge-simple")
_ASSERT_RUN_ID = DatasetRunId("surge-simple-20260313T100000000Z")
_ASSERT_EXPECTED = make_r2_prefix(_ASSERT_CONFIG_ID, _ASSERT_RUN_ID)


class TestAssertR2PrefixMatches:
    """Tests for assert_r2_prefix_matches."""

    def test_matching_prefix_does_not_raise(self) -> None:
        """No exception when the materialized prefix matches the derived value."""
        assert_r2_prefix_matches(_ASSERT_EXPECTED, _ASSERT_CONFIG_ID, _ASSERT_RUN_ID)

    def test_matching_prefix_with_explicit_root_does_not_raise(self) -> None:
        """Custom prefix_root matches when prefix was built with the same root."""
        prefix = make_r2_prefix(_ASSERT_CONFIG_ID, _ASSERT_RUN_ID, prefix_root="datasets")
        assert_r2_prefix_matches(prefix, _ASSERT_CONFIG_ID, _ASSERT_RUN_ID, prefix_root="datasets")

    def test_wrong_config_id_raises(self) -> None:
        """ValueError when config_id doesn't match what the prefix encodes."""
        with pytest.raises(ValueError, match="mismatch"):
            assert_r2_prefix_matches(
                _ASSERT_EXPECTED, DatasetConfigId("other-cfg"), _ASSERT_RUN_ID
            )

    def test_wrong_run_id_raises(self) -> None:
        """ValueError when run_id doesn't match what the prefix encodes."""
        with pytest.raises(ValueError, match="mismatch"):
            assert_r2_prefix_matches(
                _ASSERT_EXPECTED, _ASSERT_CONFIG_ID, DatasetRunId("other-20260313T100000000Z")
            )

    def test_wrong_prefix_root_raises(self) -> None:
        """ValueError when the prefix_root doesn't match the prefix's root segment."""
        with pytest.raises(ValueError, match="mismatch"):
            assert_r2_prefix_matches(
                _ASSERT_EXPECTED, _ASSERT_CONFIG_ID, _ASSERT_RUN_ID, prefix_root="train"
            )

    def test_error_message_includes_actual_and_expected(self) -> None:
        """ValueError message carries both the received and expected prefix strings."""
        bad_prefix = "data/wrong-cfg/wrong-run/"
        with pytest.raises(ValueError, match="mismatch") as exc_info:
            assert_r2_prefix_matches(bad_prefix, _ASSERT_CONFIG_ID, _ASSERT_RUN_ID)
        msg = str(exc_info.value)
        assert bad_prefix in msg
        assert _ASSERT_EXPECTED in msg
