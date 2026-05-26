#!/usr/bin/env python3
"""Print the canonical R2 URI of a materialized ``input_spec.json``.

Two modes:

- **File mode** (default): ``synth-setter-spec-uri <input_spec.json>`` reads
  the local spec file, parses it as a ``DatasetSpec``, and emits
  ``spec.r2.input_spec_uri()``.
- **Hydra-compose mode**: ``synth-setter-spec-uri --from-experiment EXP
  --run-id-override RUNID`` composes the dataset cfg via the same Hydra
  pipeline ``synth-setter-generate-dataset`` uses, with ``run_id`` pinned to
  ``RUNID``. Useful for CI cells that need to derive the URI a launcher *will*
  write, before it has written it — e.g. for per-matrix-cell validator
  pairing without the stdout-sentinel grep contract.

Both modes go through the real ``DatasetSpec`` model (so any schema drift
fails loud at this seam rather than silently in jq/sed) and emit the URI on
stdout; argv / fs / parse failures map to distinct exit codes for log scanners.

Usage::

    synth-setter-spec-uri <input_spec.json>
    synth-setter-spec-uri --from-experiment EXP --run-id-override RUNID
"""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pydantic import ValidationError

from synth_setter.pipeline.schemas.spec import DatasetSpec

# Distinct exit codes so a GitHub Actions log scanner (or a human reading the
# step output) can tell argv / fs / parse failures apart without grepping the
# stderr message text.
_EXIT_USAGE = 1
_EXIT_MISSING_FILE = 2
_EXIT_INVALID_SPEC = 3

_USAGE = (
    "Usage: synth-setter-spec-uri <input_spec.json>\n"
    "   or: synth-setter-spec-uri --from-experiment EXP --run-id-override RUNID\n"
)

# Composed-config keys that aren't ``DatasetSpec`` fields. Mirrors
# ``cli.generate_dataset._NON_SPEC_KEYS`` — keep in sync if either side adds
# a new top-level cfg sub-tree (interpolation source, dispatch group, etc.).
_NON_SPEC_CFG_KEYS: tuple[str, ...] = (
    "data",
    "paths",
    "hydra",
    "run_name",
    "skypilot_launch",
)


def compute_spec_uri(spec_path: Path) -> str:
    """Read ``spec_path`` and return the spec's canonical input_spec R2 URI.

    :param spec_path: Local path to a materialized ``input_spec.json``.
    :returns: ``spec.r2.input_spec_uri()`` —
        ``r2://<bucket>/<prefix>input_spec.json`` URI string.
    """
    spec = DatasetSpec.model_validate_json(spec_path.read_text())
    return spec.r2.input_spec_uri()


def compute_spec_uri_from_hydra(experiment: str, run_id_override: str) -> str:
    """Hydra-compose the dataset cfg with ``run_id`` pinned, return the canonical URI.

    Uses ``initialize_config_module("synth_setter.configs")`` + the same
    ``compose(config_name="dataset", ...)`` call ``synth-setter-generate-dataset``
    uses, plus a ``+run_id=<value>`` override. Pinning ``run_id`` suppresses the
    ``_default_run_id`` factory (which would otherwise sample ``created_at`` and
    produce a non-deterministic URI), and ``r2.prefix`` derivation is a pure
    function of ``(prefix_root, task_name, run_id)`` — so the resulting URI is
    fully determined by ``(experiment, run_id_override)``.

    Exceptions from Hydra ``compose`` (unknown experiment, malformed override)
    and from ``DatasetSpec`` construction propagate to the caller; the CLI
    layer collapses both onto ``_EXIT_INVALID_SPEC`` for log scanners.

    :param experiment: Hydra experiment name (e.g. ``generate_dataset/smoke-shard``).
    :param run_id_override: Cell-specific run_id; surfaces in the URI as the
        ``<run_id>`` path segment.
    :returns: ``r2://<bucket>/<prefix>input_spec.json`` URI string.
    :raises TypeError: composed cfg's top level is not a mapping.
    """
    overrides = [f"experiment={experiment}", f"+run_id={run_id_override}"]
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=overrides)
    # Programmatic compose leaves ${hydra:runtime.output_dir} unset; pin
    # paths.* with placeholders so resolve() doesn't trip — values are
    # irrelevant to URI derivation, which depends only on task_name + run_id
    # + r2.bucket.
    cfg.paths.root_dir = "."
    cfg.paths.output_dir = "."
    cfg.paths.work_dir = "."
    raw: object = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"composed config is not a mapping: {type(raw).__name__}")
    spec_kwargs = {
        k: v for k, v in raw.items() if isinstance(k, str) and k not in _NON_SPEC_CFG_KEYS
    }
    return DatasetSpec(**spec_kwargs).r2.input_spec_uri()


def _parse_hydra_argv(argv: list[str]) -> tuple[str, str] | None:
    """Extract ``(experiment, run_id_override)`` from argv when both flags are set.

    Accepts both ``--flag VALUE`` and ``--flag=VALUE`` forms. Returns ``None``
    when *neither* flag is present (caller falls through to file mode).
    Returns ``("__usage__", "")`` sentinel when *some* flag is present but the
    pair is incomplete or has stray args, signalling a usage error.

    :param argv: ``sys.argv[1:]`` slice.
    :returns: ``(experiment, run_id_override)`` on a complete pair;
        ``("__usage__", "")`` on a malformed invocation;
        ``None`` when neither flag is present.
    """
    # Exact-name match (plus the ``=value`` form) so future flags like
    # ``--from-experiment-source`` can't accidentally route through this parser.
    hydra_flags = {"--from-experiment", "--run-id-override"}
    eq_prefixes = ("--from-experiment=", "--run-id-override=")
    if not any(a in hydra_flags or a.startswith(eq_prefixes) for a in argv):
        return None

    experiment: str | None = None
    run_id_override: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--from-experiment" and i + 1 < len(argv):
            experiment = argv[i + 1]
            i += 2
        elif arg.startswith("--from-experiment="):
            experiment = arg.split("=", 1)[1]
            i += 1
        elif arg == "--run-id-override" and i + 1 < len(argv):
            run_id_override = argv[i + 1]
            i += 2
        elif arg.startswith("--run-id-override="):
            run_id_override = arg.split("=", 1)[1]
            i += 1
        else:
            return ("__usage__", "")
    if not experiment or not run_id_override:
        return ("__usage__", "")
    return (experiment, run_id_override)


def main() -> None:
    """CLI entry: file mode or Hydra-compose mode (see module docstring)."""
    argv = sys.argv[1:]

    hydra_args = _parse_hydra_argv(argv)
    if hydra_args is not None:
        experiment, run_id_override = hydra_args
        if experiment == "__usage__":
            sys.stderr.write(_USAGE)
            sys.exit(_EXIT_USAGE)
        try:
            uri = compute_spec_uri_from_hydra(experiment, run_id_override)
        # Hydra's compose can raise a wide set of types (MissingConfigException,
        # OmegaConfBaseException, OverridesParser errors, etc.). Catch broadly
        # and collapse to one stderr line — distinguishing them in the CLI buys
        # nothing the exit code doesn't already encode.
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"error: compose failed for experiment {experiment!r}: {exc}\n")
            sys.exit(_EXIT_INVALID_SPEC)
        sys.stdout.write(uri + "\n")
        return

    if len(argv) != 1:
        sys.stderr.write(_USAGE)
        sys.exit(_EXIT_USAGE)
    spec_path = Path(argv[0])
    if not spec_path.is_file():
        sys.stderr.write(f"error: spec file not found: {spec_path}\n")
        sys.exit(_EXIT_MISSING_FILE)
    try:
        uri = compute_spec_uri(spec_path)
    except (OSError, ValueError, ValidationError) as exc:
        # ValueError covers Pydantic's JSON decode error; OSError covers
        # read-time fs failures (permission denied, mid-read truncation).
        # Collapse the traceback into one stderr line so the GitHub Actions
        # step output is interpretable at a glance.
        sys.stderr.write(f"error: failed to parse spec {spec_path}: {exc}\n")
        sys.exit(_EXIT_INVALID_SPEC)
    sys.stdout.write(uri + "\n")


if __name__ == "__main__":
    main()
