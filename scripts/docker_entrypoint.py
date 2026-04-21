#!/usr/bin/env python
"""Docker entrypoint — click group with per-mode spec parsing.

The container's runtime CLI. Each spec-taking subcommand deserializes its
``--spec`` into a mode-specific pydantic model at the container boundary
(parse-don't-validate), then hands off to the downstream.

Subcommands:

  idle
    ``exec sleep infinity`` — keeps the container alive for ``docker exec``.

  passthrough ARGV...
    ``exec ARGV`` — run an arbitrary command with container's PID 1 replaced.
    Errors if no ARGV given (prevents silent no-op containers).

  generate_dataset --spec <path>
    Parse <path> as a DatasetPipelineSpec and call
    ``pipeline.entrypoints.generate_dataset.run(spec)`` in-process.

  render_eval --spec <path>
    Placeholder — raises NotImplementedError (tracked in #410).

  train --spec <path>
    Placeholder — raises NotImplementedError (tracked in #409).

See also:
  docs/reference/docker-spec.md — full runtime spec for the container.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import cast

import click
from pydantic import VERSION as _PYDANTIC_VERSION
from pydantic import BaseModel, ValidationError

from pipeline.entrypoints.generate_dataset import run
from pipeline.schemas.spec import DatasetPipelineSpec

if not _PYDANTIC_VERSION.startswith("2."):
    raise RuntimeError(f"docker_entrypoint requires pydantic v2, got {_PYDANTIC_VERSION}")

logger = logging.getLogger("docker_entrypoint")

# Maps subcommand name -> pydantic model used to parse its --spec payload.
# render_eval and train are deliberately absent; they gain entries when their
# concrete spec types exist.
_MODE_SPEC_TYPES: dict[str, type[BaseModel]] = {
    "generate_dataset": DatasetPipelineSpec,
}


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Docker entrypoint dispatch."""
    # Click's default for groups is to print help and exit 0 when invoked
    # without a subcommand. That's a silent-success footgun for a container
    # entrypoint: `docker run <image>` would start and exit cleanly having
    # done nothing. Match the bash entrypoint's fail-loud behavior on
    # unset MODE by raising a usage error instead.
    if ctx.invoked_subcommand is None:
        raise click.UsageError("Missing subcommand. Run with --help to list available modes.")


def _exec_or_click_error(program: str, argv: list[str]) -> None:
    """Exec ``program`` with ``argv`` or raise ClickException on exec failure.

    ``os.execvp`` raises ``OSError`` / ``FileNotFoundError`` when the target
    binary can't be found or executed (missing from PATH, permission denied,
    etc.). Surfacing that as a raw Python traceback makes container logs
    noisy and orchestrator-unfriendly — convert to a click error so callers
    see a clean message and a non-zero exit. A richer exit-code contract for
    execvp failures is deferred — see follow-up tracking issue.
    """
    try:
        os.execvp(program, argv)
    except OSError as exc:
        logger.error("exec failed for %s: %s", program, exc)
        raise click.ClickException(f"Unable to exec {program!r}: {exc}") from exc


@cli.command()
def idle() -> None:
    """Keep the container alive indefinitely (``exec sleep infinity``)."""
    logger.info("Entering idle mode — exec sleep infinity.")
    _exec_or_click_error("sleep", ["sleep", "infinity"])


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def passthrough(args: tuple[str, ...]) -> None:
    """Exec ARGV.

    Errors if no command was given.
    """
    if not args:
        raise click.ClickException(
            "passthrough requires a command to exec (got no trailing argv)."
        )
    logger.info("Entering passthrough mode — exec %s", args[0])
    _exec_or_click_error(args[0], list(args))


@cli.command("generate_dataset")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSON-serialized DatasetPipelineSpec.",
)
def generate_dataset(spec_path: Path) -> None:
    """Parse --spec into DatasetPipelineSpec and run the generate pipeline in-process."""
    logger.info("Entering generate_dataset mode — spec=%s", spec_path)
    spec_type = _MODE_SPEC_TYPES["generate_dataset"]
    spec = _parse_spec(spec_path, spec_type)
    run(cast(DatasetPipelineSpec, spec))


@cli.command("render_eval")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSON-serialized RenderEvalSpec (not yet defined).",
)
def render_eval(spec_path: Path) -> None:
    """Placeholder — fails loudly until #410 lands render_eval.

    Uses ``ClickException`` rather than ``NotImplementedError`` so click's
    standalone driver prints a clean ``Error: ...`` line and exits non-zero
    instead of dumping a Python traceback into container logs.
    """
    logger.error("render_eval invoked but not implemented (see #410); spec=%s", spec_path)
    raise click.ClickException("render_eval not implemented; tracked in #410")


@cli.command("train")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSON-serialized TrainSpec (not yet defined).",
)
def train(spec_path: Path) -> None:
    """Placeholder — fails loudly until #409 lands train.

    Uses ``ClickException`` rather than ``NotImplementedError`` so click's
    standalone driver prints a clean ``Error: ...`` line and exits non-zero
    instead of dumping a Python traceback into container logs.
    """
    logger.error("train invoked but not implemented (see #409); spec=%s", spec_path)
    raise click.ClickException("train not implemented; tracked in #409")


def _parse_spec(spec_path: Path, spec_type: type[BaseModel]) -> BaseModel:
    """Deserialize ``spec_path`` as ``spec_type``; surface read + validation failures."""
    try:
        spec_text = spec_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        logger.error("Spec read failed for %s: %s", spec_path, exc)
        raise click.ClickException(f"Unable to read spec at {spec_path}: {exc}") from exc

    try:
        return spec_type.model_validate_json(spec_text)
    except ValidationError as exc:
        logger.error("Spec validation failed for %s: %s", spec_path, exc)
        raise click.ClickException(f"Invalid spec at {spec_path}: {exc}") from exc


def main() -> None:
    """Configure logging and dispatch to the click group."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli()


if __name__ == "__main__":
    main()
