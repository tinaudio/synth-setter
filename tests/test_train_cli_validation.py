"""End-to-end guard that the train CLI validates its composed config.

Drives the real ``train.main()`` entrypoint (Hydra-decorated) with a malformed
override and asserts the CLI-boundary validation rejects it before any
datamodule/model/trainer is built — the behaviour wired into ``cli/train.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_setter.cli.train import main as train_main


def test_train_main_rejects_negative_seed_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``train.main()`` fails fast on ``seed=-1`` (rejected by Lightning at runtime).

    The boundary validation is ``main()``'s first step, so this raises a
    ``ValidationError`` without instantiating anything or starting a fit.

    :param tmp_path: Hosts ``PROJECT_ROOT`` so no real workspace is touched.
    :param monkeypatch: Points ``PROJECT_ROOT`` + ``sys.argv`` at the test args.
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
