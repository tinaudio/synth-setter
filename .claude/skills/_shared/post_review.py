"""Compatibility wrapper for the shared review-posting helper."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    helper = (
        Path(__file__).resolve().parents[3] / "agent" / "skills" / "_shared" / "post_review.py"
    )
    runpy.run_path(str(helper), run_name="__main__")
