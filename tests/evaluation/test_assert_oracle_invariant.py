"""Pin the log and metrics.json parsers used by the workflow gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_setter.evaluation.assert_oracle_invariant import (
    main,
    parse_metric_from_json,
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


def test_main_exits_nonzero_on_no_args() -> None:
    """Argparse rejects the call with no source argument; CLI exits non-zero."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_json_parser_extracts_metric() -> None:
    """metrics.json key lookup returns the stored float without tensor / numpy coercion."""
    payload = {"test/param_mse": 0.0, "test/per_param_mse": [0.0, 0.0, 0.0, 0.0]}
    assert parse_metric_from_json(json.dumps(payload)) == 0.0


def test_json_parser_raises_when_payload_is_not_dict() -> None:
    """A JSON list at the top level isn't a metric dict and must fail the gate, not coerce."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_metric_from_json(json.dumps([1, 2, 3]))


def test_json_parser_raises_when_key_missing() -> None:
    """Missing ``test/param_mse`` key means eval never ran the test phase — surface the gap."""
    with pytest.raises(ValueError, match="not found in metrics.json"):
        parse_metric_from_json(json.dumps({"val/param_mse": 0.0}))


def test_json_parser_raises_when_value_non_numeric() -> None:
    """A non-numeric metric (e.g. NaN serialized as the string ``"NaN"``) is a parse failure."""
    with pytest.raises(ValueError, match="non-numeric"):
        parse_metric_from_json(json.dumps({"test/param_mse": "NaN"}))


def test_main_metrics_json_mode_exits_zero_on_zero_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--metrics-json`` happy path: CLI returns 0 and reports the source label.

    :param tmp_path: Holds the JSON fixture.
    :param capsys: Captures the PASS line for source-label assertion.
    """
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"test/param_mse": 0.0}))
    assert main(["--metrics-json", str(metrics)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "metrics.json" in out


def test_main_metrics_json_mode_exits_one_on_nonzero_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--metrics-json`` failure path: CLI returns 1 and FAIL is tagged with the source.

    :param tmp_path: Holds the JSON fixture.
    :param capsys: Captures the FAIL line for source-label assertion.
    """
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"test/param_mse": 0.0001}))
    assert main(["--metrics-json", str(metrics)]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "metrics.json" in err
