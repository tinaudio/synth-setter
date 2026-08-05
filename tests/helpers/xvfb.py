"""Xvfb failure helpers for renderer-dispatch tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def install_failing_xvfb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install an Xvfb executable that records and rejects every launch.

    :param tmp_path: Scratch root for the executable and call marker.
    :param monkeypatch: Prepends the executable directory to ``PATH``.
    :returns: Marker written only when the headless wrapper starts Xvfb.
    """
    bin_dir = tmp_path / "failing-x-bin"
    bin_dir.mkdir()
    marker = tmp_path / "xvfb-called"
    xvfb = bin_dir / "Xvfb"
    xvfb.write_text(
        '#!/bin/bash\nset -euo pipefail\nprintf called > "$FAILING_XVFB_MARKER"\nexit 1\n'
    )
    xvfb.chmod(0o755)
    monkeypatch.setenv("FAILING_XVFB_MARKER", str(marker))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("XVFB_BOOTSTRAP_ATTEMPTS", "1")
    return marker
