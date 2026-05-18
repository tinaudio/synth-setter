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
# The schemas imports immediately below transitively require those extras
# (griffe-pydantic introspects model_fields at render time), so the skip
# must precede them.
pytest.importorskip("mkdocs", reason="docs extras not installed; install with -e '.[docs]'")

from synth_setter.schemas.callbacks_config import CallbackInstance
from synth_setter.schemas.data_config import DataConfig
from synth_setter.schemas.extras_config import ExtrasConfig
from synth_setter.schemas.logger_config import LoggerInstance
from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig, SchedulerConfig
from synth_setter.schemas.paths_config import PathsConfig
from synth_setter.schemas.train_config import TrainConfig
from synth_setter.schemas.trainer_config import TrainerConfig

# End-to-end mkdocs build is heavyweight (subprocess + mkdocs + mkdocstrings +
# griffe + every schemas module imported); marker keeps `make test-fast` fast.
pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Maps each config-reference page to the pydantic class(es) whose typed fields
# it documents. RootModel wrappers (CallbacksConfig, LoggerConfig) expose only
# `root`; the per-instance class is documented instead.
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


assert _PAGE_TO_MODELS, "_PAGE_TO_MODELS is empty — config-reference page map is missing"
assert _expected_field_anchors(), "_expected_field_anchors() is empty — no fields to verify"


def _anchor_pattern(model: type, field_name: str) -> re.Pattern[str]:
    """Build the regex matching the heading-anchor ``id`` mkdocstrings emits for a field.

    Targeting the structural anchor (``id="module.Class.field"``) instead of a
    bare substring avoids false positives from short field names appearing in
    HTML/CSS/JS context (``lr``, ``test``, ``train``).

    :param model: The pydantic class whose field anchor we expect to find.
    :param field_name: The field attribute name on ``model``.
    :returns: Compiled regex anchored on ``id="<module>.<Class>.<field>"``.
    :rtype: re.Pattern[str]
    """
    fqn = re.escape(f"{model.__module__}.{model.__name__}.{field_name}")
    return re.compile(rf'id="{fqn}"')


@pytest.fixture(scope="session")
def built_site(  # noqa: DOC101,DOC103,DOC201,DOC203
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run ``mkdocs build --strict`` once per session and return the site dir."""
    site_dir = tmp_path_factory.mktemp("mkdocs-site")
    try:
        subprocess.run(  # noqa: S603
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
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"mkdocs build --strict failed (exit {exc.returncode})\n"
            f"--- stdout ---\n{exc.stdout}\n--- stderr ---\n{exc.stderr}"
        )
    return site_dir


@pytest.fixture(scope="session")
def page_html_cache(built_site: Path) -> dict[str, str]:  # noqa: DOC101,DOC103,DOC201,DOC203
    """Memoised HTML text for each config-reference page."""
    return {page: (built_site / page / "index.html").read_text() for page in _PAGE_TO_MODELS}


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
    page_html_cache: dict[str, str], page: str, model: type, field: str
) -> None:
    """Every typed field on every documented model gets its own anchor heading."""
    assert page in page_html_cache, f"{page} missing from built site — see test_docs_page_emitted"
    page_html = page_html_cache[page]
    pattern = _anchor_pattern(model, field)
    assert pattern.search(page_html), (
        f"anchor {pattern.pattern!r} missing from {page} — "
        f"{model.__name__}.{field} pydantic field and docs page have drifted"
    )


def test_docs_train_config_renders_field_description(  # noqa: DOC101,DOC103
    page_html_cache: dict[str, str],
) -> None:
    """``TrainConfig.task_name``'s ``Field(description=...)`` text renders in the page."""
    description = TrainConfig.model_fields["task_name"].description
    assert description, "TrainConfig.task_name has no Field(description=...) to spot-check"
    page_html = page_html_cache["config_reference/train_config"]
    assert description in page_html, (
        f"TrainConfig.task_name description {description!r} missing from rendered page — "
        f"mkdocstrings emitted the anchor but dropped the Field(description=...) body"
    )
