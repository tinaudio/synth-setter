"""Integration tests for the mkdocs documentation build.

Runs ``mkdocs build --strict`` in a tmp directory and asserts the rendered
HTML for each auto-generated config-reference page contains the field
anchors mkdocstrings emits for every typed field on the corresponding
pydantic model. A model field rename / removal without a doc update is
caught here at PR time rather than silently dropping from the published
site.

Expected field names are derived from ``model.model_fields`` at test time
(not hard-coded) so a schema rename surfaces as a missing anchor in the
rendered HTML without anyone having to remember to update a parallel list.

``mkdocs`` is in the ``[docs]`` optional dependency group; the whole module
skips cleanly when the import fails so a developer with only ``[dev]``
installed can still run the suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Skip the whole module if the docs extras are not installed. mkdocs is the
# minimum surface; the strict build wires in mkdocs-material, mkdocstrings,
# and griffe-pydantic, which are pinned together in [project.optional-deps].
pytest.importorskip("mkdocs", reason="docs extras not installed; install with -e '.[docs]'")

from synth_setter.schemas.callbacks_config import CallbackInstance
from synth_setter.schemas.data_config import DataConfig
from synth_setter.schemas.extras_config import ExtrasConfig
from synth_setter.schemas.logger_config import LoggerInstance
from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig, SchedulerConfig
from synth_setter.schemas.paths_config import PathsConfig
from synth_setter.schemas.train_config import TrainConfig
from synth_setter.schemas.trainer_config import TrainerConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pages map to the pydantic classes whose typed fields they document. The
# test derives the expected field set from ``model_fields`` at runtime, so a
# rename or removal on the schema shows up as a missing anchor in the
# rendered HTML rather than as a manually-updated constant going stale.
# ``mkdocs build --strict`` already validates the dataset-spec page renders
# without warnings, so it doesn't need an entry here.
#
# ``CallbacksConfig`` and ``LoggerConfig`` are ``RootModel`` wrappers around
# ``dict[str, CallbackInstance | LoggerInstance]``; their only declared field
# is ``root`` and griffe-pydantic doesn't surface it as an anchor heading.
# The callbacks/logger pages also render the inner instance model, so the
# per-field assertions exercise that class instead.
_PAGE_TO_MODELS: dict[str, tuple[type, ...]] = {
    "config_reference/train_config": (TrainConfig,),
    "config_reference/data_config": (DataConfig,),
    "config_reference/model_config": (ModelConfig, OptimizerConfig, SchedulerConfig),
    "config_reference/callbacks_config": (CallbackInstance,),
    "config_reference/logger_config": (LoggerInstance,),
    "config_reference/trainer_config": (TrainerConfig,),
    "config_reference/paths_config": (PathsConfig,),
    "config_reference/extras_config": (ExtrasConfig,),
}


def _expected_field_anchors() -> list[tuple[str, type, str]]:  # noqa: DOC201,DOC203
    """Build ``(page, model, field_name)`` tuples for every typed field on every model."""
    triples: list[tuple[str, type, str]] = []
    for page, models in _PAGE_TO_MODELS.items():
        for model in models:
            for field_name in model.model_fields:
                triples.append((page, model, field_name))
    return triples


def _anchor_pattern(model: type, field_name: str) -> re.Pattern[str]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Build the regex matching the heading-anchor ``id`` mkdocstrings emits for a field.

    Targeting the structural anchor (``id="module.Class.field"``) instead of a
    bare substring avoids false positives from short field names appearing in
    HTML/CSS/JS context (``lr``, ``test``, ``train``).
    """
    fqn = re.escape(f"{model.__module__}.{model.__name__}.{field_name}")
    return re.compile(rf'id="{fqn}"')


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


@pytest.mark.parametrize("page", list(_PAGE_TO_MODELS))
def test_docs_page_emitted(built_site: Path, page: str) -> None:  # noqa: DOC101,DOC103
    """Each config-reference page exists in the built site."""
    page_html = built_site / page / "index.html"
    assert page_html.is_file(), f"{page_html} missing from built site"


@pytest.mark.parametrize(
    ("page", "model", "field"),
    _expected_field_anchors(),
    ids=lambda v: v if isinstance(v, str) else getattr(v, "__name__", repr(v)),
)
def test_docs_page_renders_pydantic_field_anchor(  # noqa: DOC101,DOC103
    built_site: Path, page: str, model: type, field: str
) -> None:
    """Every typed field on every documented model gets its own anchor heading."""
    # Assert the file exists before reading so a missing page surfaces as a
    # clear pytest assertion (with the page name) rather than the raw
    # FileNotFoundError that ``read_text`` would raise — the dedicated
    # ``test_docs_page_emitted`` case still owns the per-page emission check.
    page_html_path = built_site / page / "index.html"
    assert page_html_path.is_file(), (
        f"{page_html_path} missing from built site — see test_docs_page_emitted"
    )
    page_html = page_html_path.read_text(encoding="utf-8")
    pattern = _anchor_pattern(model, field)
    assert pattern.search(page_html), (
        f"anchor {pattern.pattern!r} missing from {page} — "
        f"{model.__name__}.{field} pydantic field and docs page have drifted"
    )
