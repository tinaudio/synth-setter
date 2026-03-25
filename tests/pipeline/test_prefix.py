from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pipeline.prefix import make_dataset_wandb_run_id, make_r2_prefix


class TestMakeDatasetWandbRunId:
    """Tests for make_dataset_wandb_run_id."""

    def test_make_run_id_fixed_timestamp_format(self):
        """Fixed UTC timestamp produces the expected run ID string."""
        ts = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        result = make_dataset_wandb_run_id("surge-simple-480k-10k", timestamp=ts)
        assert result == "surge-simple-480k-10k-20260313T100000Z"

    def test_make_run_id_default_timestamp_is_utc(self):
        """Default (no timestamp) produces an ID ending with Z containing today's date."""
        result = make_dataset_wandb_run_id("test-config")
        assert result.endswith("Z")
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert today in result

    def test_make_run_id_rejects_naive_timestamp(self):
        """Naive datetime (no tzinfo) raises ValueError."""
        import pytest

        naive = datetime(2026, 3, 13, 10, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            make_dataset_wandb_run_id("cfg", timestamp=naive)

    def test_make_run_id_rejects_non_utc_timezone(self):
        """Non-UTC timezone raises ValueError."""
        import pytest

        non_utc = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone(timedelta(hours=5)))
        with pytest.raises(ValueError, match="must be UTC"):
            make_dataset_wandb_run_id("cfg", timestamp=non_utc)

    def test_make_run_id_deterministic_same_inputs(self):
        """Same arguments produce the same result."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        a = make_dataset_wandb_run_id("cfg", timestamp=ts)
        b = make_dataset_wandb_run_id("cfg", timestamp=ts)
        assert a == b

    def test_make_run_id_seconds_precision(self):
        """Timestamps one second apart produce different IDs."""
        ts1 = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 13, 10, 0, 1, tzinfo=timezone.utc)
        id1 = make_dataset_wandb_run_id("cfg", timestamp=ts1)
        id2 = make_dataset_wandb_run_id("cfg", timestamp=ts2)
        assert id1 != id2


class TestMakeR2Prefix:
    """Tests for make_r2_prefix."""

    def test_make_r2_prefix_format(self):
        """Prefix matches the expected data/<config_id>/<run_id>/ pattern."""
        result = make_r2_prefix(
            "surge-simple-480k-10k",
            "surge-simple-480k-10k-20260313T100000Z",
        )
        assert result == "data/surge-simple-480k-10k/surge-simple-480k-10k-20260313T100000Z/"

    def test_make_r2_prefix_trailing_slash(self):
        """Prefix always ends with a trailing slash."""
        result = make_r2_prefix("a", "b")
        assert result.endswith("/")
