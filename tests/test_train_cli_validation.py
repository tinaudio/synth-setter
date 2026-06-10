"""End-to-end guard that the train CLI validates its composed config.

Drives the real ``train.main()`` entrypoint (Hydra-decorated) with a malformed
override and asserts the CLI-boundary validation rejects it before any
datamodule/model/trainer is built — the behaviour wired into ``cli/train.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from pydantic import ValidationError

from synth_setter.cli.train import main as train_main


@pytest.fixture
def restore_hydra_and_cwd() -> Iterator[None]:
    """Restore the cwd and clear Hydra's singleton after a ``@hydra.main`` call.

    ``@hydra.main`` startup can ``chdir`` and initialize ``GlobalHydra`` even
    when the body raises, so this teardown runs unconditionally to keep the
    failure from leaking into sibling tests.
    """
    original_cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()


def test_train_main_rejects_negative_seed_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_hydra_and_cwd: None,
) -> None:
    """``train.main()`` fails fast on ``seed=-1`` (rejected by Lightning at runtime).

    The boundary validation is ``main()``'s first step, so this raises a
    ``ValidationError`` without instantiating anything or starting a fit.

    :param tmp_path: Hosts ``PROJECT_ROOT`` so no real workspace is touched.
    :param monkeypatch: Points ``PROJECT_ROOT`` + ``sys.argv`` at the test args.
    :param restore_hydra_and_cwd: Resets cwd + Hydra's singleton on teardown.
    """
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    # ``run_name`` is interpolated into Hydra's ``run.dir`` template, so it must
    # be set for ``@hydra.main`` startup to resolve before validation runs.
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "datamodule=ksin",
            "model=ffn",
            "trainer=cpu",
            "+run_name=cli-validation",
            "seed=-1",
        ],
    )

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        train_main()
