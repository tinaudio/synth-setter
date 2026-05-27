"""Parse a `synth-setter-eval` log for `test/param_mse` and assert it is exactly zero.

Used by the `Generate + Finalize + Oracle-Eval Dataset (inline)` workflow to turn Lightning's
printed metrics table into a hard pass/fail gate — the oracle's load-bearing invariant is `pred ==
target` (and therefore `param_mse == 0.0`), so any other value means generate or finalize corrupted
the parameter-array round-trip.
"""

from __future__ import annotations

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


def main(argv: list[str] | None = None) -> int:
    """CLI: read the log path from argv, parse, and exit non-zero on failure.

    :param argv: Argument list excluding the program name; defaults to
        ``sys.argv[1:]``.
    :returns: 0 when ``test/param_mse == 0.0`` exactly; 1 otherwise (also when
        the metric is missing, non-numeric, or NaN).
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: assert_oracle_invariant <log-path>", file=sys.stderr)  # noqa: T201
        return 1
    log_path = Path(args[0])
    try:
        value = parse_metric_from_log(log_path.read_text())
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    if value != 0.0:
        print(  # noqa: T201
            f"FAIL: test/param_mse={value!r}; oracle requires exact 0.0",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: test/param_mse={value}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
