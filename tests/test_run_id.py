"""Tests for the shared run-id convention in ``synth_setter.run_id``."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from synth_setter.run_id import make_wandb_run_id


class TestMakeWandbRunId:
    """The shared ``{config_id}-{timestamp}`` format used by data, training, and eval runs."""

    def test_fixed_utc_timestamp_produces_expected_id(self) -> None:
        """A whole-second UTC timestamp yields ``<config_id>-YYYYMMDDTHHMMSS000Z``."""
        ts = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone.utc)
        assert make_wandb_run_id("flow_simple", timestamp=ts) == "flow_simple-20260313T100000000Z"

    def test_microseconds_render_as_three_digit_millis(self) -> None:
        """Sub-second precision is truncated to a 3-digit millisecond field."""
        ts = datetime(2026, 5, 3, 13, 36, 33, 456789, tzinfo=timezone.utc)
        assert make_wandb_run_id("cfg", timestamp=ts) == "cfg-20260503T133633456Z"

    def test_naive_timestamp_raises(self) -> None:
        """A timezone-naive datetime is rejected."""
        with pytest.raises(ValueError, match="timezone-aware"):
            make_wandb_run_id("cfg", timestamp=datetime(2026, 3, 13, 10, 0, 0))

    def test_non_utc_timestamp_raises(self) -> None:
        """A non-UTC timezone is rejected."""
        non_utc = datetime(2026, 3, 13, 10, 0, 0, tzinfo=timezone(timedelta(hours=5)))
        with pytest.raises(ValueError, match="must be UTC"):
            make_wandb_run_id("cfg", timestamp=non_utc)

    def test_default_timestamp_prefixes_config_id(self) -> None:
        """With no timestamp the id is ``<config_id>-`` plus a canonical UTC stamp."""
        assert re.fullmatch(r"flow_simple-\d{8}T\d{9}Z", make_wandb_run_id("flow_simple"))
