"""Integration tests for the mkdocs documentation build.

Runs ``mkdocs build --strict`` in a tmp directory and asserts the rendered
HTML for each auto-generated config-reference page contains the field names
declared on the corresponding pydantic model. A model field rename / removal
without a doc update is caught here at PR time rather than silently dropping
from the published site.

``mkdocs`` is in the ``[docs]`` optional dependency group; the whole module
skips cleanly when the import fails so a developer with only ``[dev]``
installed can still run the suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Skip the whole module if the docs extras are not installed. mkdocs is the
# minimum surface; the strict build wires in mkdocs-material, mkdocstrings,
# and griffe-pydantic, which are pinned together in [project.optional-deps].
pytest.importorskip("mkdocs", reason="docs extras not installed; install with -e '.[docs]'")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Field names that must appear in the rendered HTML for each page. Sourced
# from the pydantic models (``TrainConfig``, ``ModelConfig`` family) so a
# rename on the model surfaces here as a doc-drift failure instead of
# silently dropping from the published site. ``mkdocs build --strict``
# already validates the dataset-spec page renders without warnings, so it
# does not need a field-list entry here.
_EXPECTED_FIELDS_BY_PAGE: dict[str, tuple[str, ...]] = {
    "config_reference/train_config": (
        "task_name",
        "tags",
        "train",
        "test",
        "ckpt_path",
        "seed",
        "optimized_metric",
        "watch_gradients",
    ),
    "config_reference/model_config": (
        "_target_",
        "optimizer",
        "scheduler",
        "compile",
        "lr",
        "weight_decay",
    ),
}


@pytest.fixture(scope="session")
def built_site(  # noqa: DOC101,DOC103,DOC201,DOC203
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run ``mkdocs build --strict`` once per session and return the site dir."""
    site_dir = tmp_path_factory.mktemp("mkdocs-site")
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(site_dir),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"mkdocs build --strict failed (exit {result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return site_dir


@pytest.mark.parametrize("page", list(_EXPECTED_FIELDS_BY_PAGE))
def test_docs_page_emitted(built_site: Path, page: str) -> None:  # noqa: DOC101,DOC103
    """Each config-reference page exists in the built site."""
    page_html = built_site / page / "index.html"
    assert page_html.is_file(), f"{page_html} missing from built site"


@pytest.mark.parametrize(
    ("page", "field"),
    [(page, field) for page, fields in _EXPECTED_FIELDS_BY_PAGE.items() for field in fields],
)
def test_docs_page_renders_pydantic_field(  # noqa: DOC101,DOC103
    built_site: Path, page: str, field: str
) -> None:
    """Each pydantic-model field appears verbatim in its page's HTML."""
    page_html = (built_site / page / "index.html").read_text(encoding="utf-8")
    assert field in page_html, (
        f"field {field!r} missing from {page} — pydantic model and docs page have drifted"
    )
