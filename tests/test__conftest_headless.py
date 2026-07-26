"""Tests for Linux headless fixture subprocess boundaries."""

import sys
from pathlib import Path

import pytest

from tests import conftest as root_conftest


@pytest.mark.skipif(sys.platform != "linux", reason="headless wrapper is Linux-only")
def test_render_smoke_train_subprocess_nonexecutable_wrapper_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A synced wrapper without its executable bit still launches through bash.

    :param monkeypatch: Replaces the wrapper with a non-executable real script.
    :param tmp_path: Provides the wrapper and expected output paths.
    """
    wrapper = tmp_path / "headless-wrapper.sh"
    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\ntouch "$3"\n')
    wrapper.chmod(0o644)
    output_path = tmp_path / "train.lance"
    monkeypatch.setattr(root_conftest, "VST_HEADLESS_WRAPPER", str(wrapper))

    root_conftest._render_smoke_train_subprocess(output_path, "surge_4", num_samples=1)

    assert output_path.exists()
