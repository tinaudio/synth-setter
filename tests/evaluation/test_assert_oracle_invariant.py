"""Pin the log-parser used by the generate-finalize-oracle-eval workflow's gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_setter.evaluation.assert_oracle_invariant import (
    main,
    parse_metric_from_log,
)

_LIGHTNING_TABLE_ZERO = """\
Testing DataLoader 0: 100%|██████████| 1/1 [00:00<00:00,  6.30it/s]
────────────────────────────────────────────────────────────────────────────
      Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────
     test/param_mse                  0.0
────────────────────────────────────────────────────────────────────────────
"""


_LIGHTNING_TABLE_NONZERO = """\
     test/param_mse                  0.0001234
"""


def test_parser_extracts_zero_from_lightning_table() -> None:
    """Real Lightning table layout parses to 0.0 — pins the workflow gate's happy path."""
    assert parse_metric_from_log(_LIGHTNING_TABLE_ZERO) == 0.0


def test_parser_extracts_nonzero_decimal() -> None:
    """Decimal metric values parse correctly so a non-zero MSE surfaces in the gate."""
    assert parse_metric_from_log(_LIGHTNING_TABLE_NONZERO) == pytest.approx(0.0001234)


def test_parser_uses_last_match_when_metric_appears_multiple_times() -> None:
    """Last match wins so the final-epoch aggregate beats per-batch progress lines."""
    text = "progress test/param_mse = 0.42\nfinal test/param_mse 0.0\n"
    assert parse_metric_from_log(text) == 0.0


def test_parser_raises_when_metric_missing() -> None:
    """Subprocess that exited before logging the metric must fail the gate, not pass silently."""
    with pytest.raises(ValueError, match="not found"):
        parse_metric_from_log("no metric here\n")


def test_parser_raises_when_metric_line_has_no_float() -> None:
    """A metric line without a numeric value is a parse failure, not a zero match."""
    with pytest.raises(ValueError, match="no float"):
        parse_metric_from_log("test/param_mse pending\n")


def test_main_exits_zero_on_zero_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI returns 0 when the log carries the exact-zero oracle invariant.

    :param tmp_path: Holds the fixture log file.
    :param capsys: Captures the PASS line for assertion.
    """
    log = tmp_path / "eval.log"
    log.write_text(_LIGHTNING_TABLE_ZERO)
    assert main([str(log)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_main_exits_one_on_nonzero_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI returns 1 when the metric is finite-but-nonzero — the round-trip drifted.

    :param tmp_path: Holds the fixture log file.
    :param capsys: Captures the FAIL line for assertion.
    """
    log = tmp_path / "eval.log"
    log.write_text(_LIGHTNING_TABLE_NONZERO)
    assert main([str(log)]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_main_exits_one_on_missing_log(tmp_path: Path) -> None:
    """CLI returns 1 (not crash) when the log file is absent.

    :param tmp_path: Used to construct an absent path.
    """
    assert main([str(tmp_path / "missing.log")]) == 1


def test_main_exits_one_on_wrong_argc() -> None:
    """CLI returns 1 on missing or extra positional arguments rather than tracebacking."""
    assert main([]) == 1
    assert main(["a", "b"]) == 1
