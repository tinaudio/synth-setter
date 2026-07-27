"""CLI migrating legacy ``torch.compile`` checkpoints to the uncompiled key layout.

Checkpoints written before in-place compilation (#2259) carry ``_orig_mod``
path parts that strict Lightning loading rejects; this strips them.
"""

import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click
import torch

_COMPILED_KEY_PART = "_orig_mod"
# Prefix torch uses for every strict load_state_dict failure; scopes the hint to
# load errors so unrelated runtime errors mentioning _orig_mod pass through.
_STRICT_LOAD_ERROR_MARKER = "Error(s) in loading state_dict"


def _canonical_key(key: str) -> str:
    """Return a key without compile-wrapper path parts.

    :param key: State-dict key.
    :returns: Key in the uncompiled layout.
    """
    return ".".join(part for part in key.split(".") if part != _COMPILED_KEY_PART)


def strip_compile_wrapper_keys(state_dict: dict[str, object]) -> dict[str, object]:
    """Rewrite wrapper-prefixed keys to the uncompiled layout, values untouched.

    :param state_dict: Checkpoint ``state_dict`` mapping.
    :returns: Mapping with ``_orig_mod`` path parts removed from every key.
    :raises ValueError: If stripping would merge two distinct keys.
    """
    stripped: dict[str, object] = {}
    colliding: list[str] = []
    for key, value in state_dict.items():
        canonical = _canonical_key(key)
        if canonical in stripped:
            colliding.append(canonical)
        stripped[canonical] = value
    if colliding:
        raise ValueError(
            f"keys collide after stripping {_COMPILED_KEY_PART!r}: {sorted(colliding)}"
        )
    return stripped


def migrate_checkpoint(input_path: Path, output_path: Path) -> int:  # noqa: DOC503 — FileExistsError comes from open("xb"); bare re-raise after cleanup
    """Write a copy of a checkpoint with its state dict in the uncompiled layout.

    :param input_path: Legacy checkpoint containing ``_orig_mod`` keys.
    :param output_path: Destination for the migrated checkpoint; must not exist.
    :returns: Number of rewritten keys.
    :raises ValueError: If no key carries a wrapper part, or stripping collides.
    :raises FileExistsError: If the destination already exists.
    """
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    wrapped = [key for key in state_dict if _COMPILED_KEY_PART in key.split(".")]
    if not wrapped:
        raise ValueError(f"no {_COMPILED_KEY_PART!r} keys in {input_path}; nothing to migrate")
    checkpoint["state_dict"] = strip_compile_wrapper_keys(state_dict)
    # "x" reserves the destination atomically, closing the check-then-save race.
    output_file = output_path.open("xb")
    try:
        with output_file:
            torch.save(checkpoint, output_file)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return len(wrapped)


@contextmanager
def checkpoint_migration_hint(ckpt_path: object) -> Iterator[None]:
    """Re-raise strict-load failures on legacy compiled checkpoints with the fix.

    :param ckpt_path: Checkpoint path the wrapped Trainer call is loading; may be None.
    :yields: Control to the wrapped Trainer call.
    :ytype: None
    :raises RuntimeError: The original error, augmented with the migration command
        when it names ``_orig_mod`` keys.
    """
    try:
        yield
    except RuntimeError as err:
        message = str(err)
        if (
            ckpt_path is None
            or _STRICT_LOAD_ERROR_MARKER not in message
            or _COMPILED_KEY_PART not in message
        ):
            raise
        quoted_input = shlex.quote(str(ckpt_path))
        quoted_output = shlex.quote(f"{ckpt_path}.migrated")
        raise RuntimeError(
            f"Checkpoint {ckpt_path} was written by a torch.compile run before "
            f"in-place compilation and carries {_COMPILED_KEY_PART!r} keys (#2259). "
            f"Migrate it with: synth-setter-migrate-checkpoint {quoted_input} "
            f"{quoted_output} — then point ckpt_path at the migrated file."
        ) from err


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
def main(input_path: Path, output_path: Path) -> None:
    """Strip torch.compile ``_orig_mod`` key parts from a legacy checkpoint.

    :param input_path: Legacy checkpoint containing ``_orig_mod`` keys.
    :param output_path: Destination for the migrated checkpoint; must not exist.
    :raises click.ClickException: If the output exists or the checkpoint cannot migrate.
    """
    try:
        count = migrate_checkpoint(input_path, output_path)
    except FileExistsError as err:
        raise click.ClickException(f"refusing to overwrite existing {output_path}") from err
    except ValueError as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"Migrated {count} wrapped keys: {input_path} -> {output_path}")
