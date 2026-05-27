"""Assert ``test/param_mse == 0.0`` against the eval log or the metrics.json artifact.

Used by the `Generate + Finalize + Oracle-Eval Dataset (inline)` workflow as a
two-pronged gate on the oracle's load-bearing invariant (``pred == target`` →
``param_mse == 0.0``). The log assertion catches anything that lands in
Lightning's stdout table; the JSON assertion reads the structured artifact
`cli/eval.py` writes to the per-run output dir. Either failing means generate
or finalize corrupted the parameter-array round-trip.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_METRIC_KEY = "test/param_mse"
_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_metric_from_log(log_text: str, metric_key: str = _METRIC_KEY) -> float:
    """Return the float that Lightning printed next to ``metric_key`` in its final table.

    Lightning's `trainer.test()` writes a final metrics table whose rows look
    like ``| test/param_mse | 0.0 |`` (the exact box-drawing characters vary by
    version). The last occurrence of ``metric_key`` is the test-epoch value;
    earlier matches may be per-batch progress lines that don't carry the
    final aggregate.

    :param log_text: Captured stdout/stderr from the eval subprocess.
    :param metric_key: Lightning log key to look up; defaults to
        ``"test/param_mse"`` — the only key the oracle test mode emits.
    :returns: Parsed float value from the last matching line.
    :raises ValueError: ``metric_key`` not found, or no float follows it.
    """
    matches = [line for line in log_text.splitlines() if metric_key in line]
    if not matches:
        raise ValueError(f"{metric_key!r} not found in log")
    last_line = matches[-1]
    floats = _FLOAT_RE.findall(last_line)
    if not floats:
        raise ValueError(f"no float found on metric line: {last_line!r}")
    return float(floats[-1])


def parse_metric_from_json(json_text: str, metric_key: str = _METRIC_KEY) -> float:
    """Return the float at ``metric_key`` from the metrics.json artifact.

    :param json_text: Contents of the per-run ``metrics/metrics.json`` file.
    :param metric_key: Key to look up; defaults to ``"test/param_mse"``.
    :returns: Coerced ``float(payload[metric_key])``.
    :raises ValueError: Payload not a dict, key missing, or value not numeric.
    """
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError(f"metrics.json must be a JSON object; got {type(payload).__name__}")
    if metric_key not in payload:
        raise ValueError(f"{metric_key!r} not found in metrics.json (keys: {sorted(payload)})")
    value = payload[metric_key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{metric_key!r} is non-numeric in metrics.json: {value!r}")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    """CLI: assert ``test/param_mse == 0.0`` against a log file or a metrics.json artifact.

    Pass either the eval subprocess's captured log (default) or the JSON
    artifact via ``--metrics-json``. Running both back-to-back from a
    workflow asserts the invariant twice — once on the unstructured stdout
    table, once on the structured artifact — so a regression in either
    surface fails the gate.

    :param argv: Argument list excluding the program name; defaults to
        ``sys.argv[1:]``.
    :returns: 0 when ``test/param_mse == 0.0`` exactly; 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("log_path", nargs="?", type=Path, help="Path to the eval subprocess log.")
    source.add_argument(
        "--metrics-json",
        type=Path,
        help="Path to ``metrics/metrics.json`` written by ``cli/eval.py:_dump_metric_dict``.",
    )
    ns = parser.parse_args(argv)
    source_path = ns.metrics_json if ns.metrics_json is not None else ns.log_path
    source_label = "metrics.json" if ns.metrics_json is not None else "log"
    parse = parse_metric_from_json if ns.metrics_json is not None else parse_metric_from_log

    try:
        value = parse(source_path.read_text())
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL ({source_label}): {exc}", file=sys.stderr)  # noqa: T201
        return 1
    if value != 0.0:
        print(  # noqa: T201
            f"FAIL ({source_label}): test/param_mse={value!r}; oracle requires exact 0.0",
            file=sys.stderr,
        )
        return 1
    print(f"PASS ({source_label}): test/param_mse={value}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
