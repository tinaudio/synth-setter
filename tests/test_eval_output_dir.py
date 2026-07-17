"""The eval entrypoint must create ``paths.output_dir`` before extras writes into it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_eval_creates_absent_nested_output_dir_before_tag_log(tmp_path: Path) -> None:
    """A nonexistent nested ``paths.output_dir`` is created before ``tags.log`` is written.

    The run still fails later on the bogus checkpoint — the contract under test is
    only that the failure is not the pre-setup ``FileNotFoundError`` at ``tags.log``.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    output_dir = tmp_path / "absent" / "child"

    result = subprocess.run(  # noqa: S603 — sys.executable + literal overrides
        [
            sys.executable,
            "-m",
            "synth_setter.cli.eval",
            "datamodule=surge_simple",
            "model=surge_ffn",
            "trainer=cpu",
            "tags=[w7test]",
            "+run_name=w7",
            "+experiment_name=w7eval",
            f"ckpt_path={tmp_path / 'missing.ckpt'}",
            f"paths.output_dir={output_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert (output_dir / "tags.log").is_file(), result.stderr[-2000:]
    assert "w7test" in (output_dir / "tags.log").read_text()
    assert f"{output_dir / 'tags.log'}" not in result.stderr
