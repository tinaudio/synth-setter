"""Sentinel test for the deflake-mps PR self-test.

Always fails when run, so the deflake harness's failure paths
(``tmp_path_retention_policy=failed`` retention, junit failure aggregation,
summarizer fail-rate arithmetic) are exercised end-to-end on every PR that
touches ``.github/workflows/deflake-mps.yml``. Gated by
``DEFLAKE_SELF_TEST=1`` so default ``pytest tests/`` runs stay green.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DEFLAKE_SELF_TEST") != "1",
    reason="deflake sentinel — set DEFLAKE_SELF_TEST=1 to opt in (deflake-mps.yml does)",
)


def test_always_fails(tmp_path):
    """Write a sentinel marker into ``tmp_path``, then fail.

    The marker filename is asserted by the workflow's
    ``Verify artifact contents`` step to prove
    ``tmp_path_retention_policy=failed`` actually retained the dir.
    """
    (tmp_path / "deflake-self-test-marker").write_text("sentinel")
    pytest.fail("intentional sentinel failure for the deflake self-test")
