"""Tests for the legacy compiled-checkpoint migration CLI and load-error hint."""

from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from synth_setter.cli.migrate_checkpoint import checkpoint_migration_hint, main


def _legacy_checkpoint(path: Path) -> torch.nn.Linear:
    """Write a checkpoint with pre-in-place-compilation wrapper keys.

    :param path: Destination checkpoint file.
    :returns: The uncompiled network whose weights the checkpoint carries.
    """
    net = torch.nn.Linear(2, 2)
    wrapped_state = {f"net._orig_mod.{key}": value for key, value in net.state_dict().items()}
    torch.save({"state_dict": wrapped_state, "epoch": 3}, path)
    return net


def test_migrate_strips_wrapper_keys_and_output_loads_strict(tmp_path: Path) -> None:
    """The migrated checkpoint strict-loads into an uncompiled module unchanged.

    :param tmp_path: Per-test directory for checkpoint files.
    """
    input_path = tmp_path / "legacy.ckpt"
    output_path = tmp_path / "migrated.ckpt"
    source_net = _legacy_checkpoint(input_path)

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code == 0, result.output
    migrated = torch.load(output_path, map_location="cpu", weights_only=False)
    consumer = torch.nn.Linear(2, 2)
    consumer.load_state_dict(
        {key.removeprefix("net."): value for key, value in migrated["state_dict"].items()}
    )
    inputs = torch.tensor([[1.0, 2.0]])
    assert torch.equal(consumer(inputs), source_net(inputs))
    assert migrated["epoch"] == 3


def test_migrate_without_wrapper_keys_exits_nonzero(tmp_path: Path) -> None:
    """A checkpoint that is already in the uncompiled layout is not rewritten.

    :param tmp_path: Per-test directory for checkpoint files.
    """
    input_path = tmp_path / "clean.ckpt"
    output_path = tmp_path / "migrated.ckpt"
    torch.save({"state_dict": torch.nn.Linear(1, 1).state_dict()}, input_path)

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code != 0
    assert "_orig_mod" in result.output
    assert not output_path.exists()


def test_migrate_existing_output_refuses_overwrite(tmp_path: Path) -> None:
    """An existing output file is never overwritten.

    :param tmp_path: Per-test directory for checkpoint files.
    """
    input_path = tmp_path / "legacy.ckpt"
    output_path = tmp_path / "migrated.ckpt"
    _legacy_checkpoint(input_path)
    output_path.write_bytes(b"existing")

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code != 0
    assert output_path.read_bytes() == b"existing"


def test_migrate_colliding_canonical_keys_aborts(tmp_path: Path) -> None:
    """Stripping that would merge two distinct keys aborts without output.

    :param tmp_path: Per-test directory for checkpoint files.
    """
    input_path = tmp_path / "legacy.ckpt"
    output_path = tmp_path / "migrated.ckpt"
    state = {
        "net._orig_mod.weight": torch.ones(1),
        "net.weight": torch.zeros(1),
    }
    torch.save({"state_dict": state}, input_path)

    result = CliRunner().invoke(main, [str(input_path), str(output_path)])

    assert result.exit_code != 0
    assert "collide" in result.output
    assert not output_path.exists()


def test_migration_hint_augments_strict_load_error_with_command() -> None:
    """A strict state-dict failure on wrapper keys gains the migration command.

    :raises RuntimeError: Re-raised by the hint; asserted via ``pytest.raises``.
    """
    with pytest.raises(RuntimeError, match="synth-setter-migrate-checkpoint"):
        with checkpoint_migration_hint("model.ckpt"):
            raise RuntimeError(
                "Error(s) in loading state_dict for Module:\n"
                '\tUnexpected key(s) in state_dict: "net._orig_mod.weight".'
            )


def test_migration_hint_leaves_unrelated_orig_mod_errors_untouched() -> None:
    """A non-load RuntimeError that merely mentions the wrapper propagates as-is.

    :raises RuntimeError: Propagated unchanged; asserted via ``pytest.raises``.
    """
    with pytest.raises(RuntimeError, match="^compiled _orig_mod graph break$"):
        with checkpoint_migration_hint("model.ckpt"):
            raise RuntimeError("compiled _orig_mod graph break")
