from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from pipeline.schemas.prefix import (
    DatasetConfigId,
    DatasetRunId,
    make_dataset_wandb_run_id,
    make_r2_prefix,
)

FIXED_NOW = datetime(2026, 6, 15, 12, 30, 45, tzinfo=timezone.utc)


class TestMakeDatasetWandbRunId:
    """Tests for make_dataset_wandb_run_id."""

    def test_make_run_id_fixed_timestamp_format(self):
        # plumb:req-6df7c153
        # plumb:req-1e7bbada
        # plumb:req-d15a2674
        """Fixed UTC timestamp produces the expected run ID string."""
        ts = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        result = make_dataset_wandb_run_id(DatasetConfigId("surge-simple-480k-10k"), timestamp=ts)
        assert result == "surge-simple-480k-10k-20260313T100000Z"

    @patch("pipeline.schemas.prefix.datetime")
    def test_make_run_id_default_timestamp_is_utc(self, mock_datetime):
        """Default (no timestamp) uses datetime.now(UTC) and produces the expected ID."""
        mock_datetime.now.return_value = FIXED_NOW
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = make_dataset_wandb_run_id(DatasetConfigId("test-config"))
        mock_datetime.now.assert_called_once_with(timezone.utc)
        assert result == "test-config-20260615T123045Z"

    def test_make_run_id_rejects_naive_timestamp(self):
        # plumb:req-74aa845b
        """Naive datetime (no tzinfo) raises ValueError."""
        import pytest

        naive = datetime(2026, 3, 13, 10, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=naive)

    def test_make_run_id_rejects_non_utc_timezone(self):
        """Non-UTC timezone raises ValueError."""
        import pytest

        non_utc = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone(timedelta(hours=5)))
        with pytest.raises(ValueError, match="must be UTC"):
            make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=non_utc)

    def test_make_run_id_deterministic_same_inputs(self):
        """Same arguments produce the same result."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        a = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts)
        b = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts)
        assert a == b

    def test_make_run_id_seconds_precision(self):
        """Timestamps one second apart produce different IDs."""
        ts1 = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 13, 10, 0, 1, tzinfo=timezone.utc)
        id1 = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts1)
        id2 = make_dataset_wandb_run_id(DatasetConfigId("cfg"), timestamp=ts2)
        assert id1 != id2


class TestMakeR2Prefix:
    """Tests for make_r2_prefix."""

    def test_make_r2_prefix_format(self):
        # plumb:req-8f913909
        # plumb:req-04698363
        """Prefix matches the expected data/<config_id>/<run_id>/ pattern."""
        result = make_r2_prefix(
            DatasetConfigId("surge-simple-480k-10k"),
            DatasetRunId("surge-simple-480k-10k-20260313T100000Z"),
        )
        assert result == "data/surge-simple-480k-10k/surge-simple-480k-10k-20260313T100000Z/"

    def test_make_r2_prefix_trailing_slash(self):
        """Prefix always ends with a trailing slash."""
        result = make_r2_prefix(DatasetConfigId("a"), DatasetRunId("b"))
        assert result.endswith("/")
