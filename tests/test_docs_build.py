"""Integration tests for the mkdocs documentation build.

Runs ``mkdocs build --strict`` once per session and asserts each
config-reference page renders an anchor for every typed pydantic field on
its documented model. Field names are derived from ``model_fields`` at test
time so a rename surfaces as a missing anchor without a parallel list.

Whole module skips when the ``docs`` dependency-group is absent (gated on
both ``mkdocs`` and ``griffe_pydantic``).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Skip precedes the schemas imports because the docs build needs both mkdocs and
# griffe-pydantic from the ``docs`` dependency-group; gating on only one lets the
# build fail later if the other is missing.
pytest.importorskip(
    "mkdocs",
    reason="docs deps missing; run the suite with `uv pip install --group dev -e .` (dev ⊇ docs + runtime — tests/conftest.py imports the heavy runtime)",
)
pytest.importorskip(
    "griffe_pydantic",
    reason="docs deps missing; run the suite with `uv pip install --group dev -e .` (dev ⊇ docs + runtime — tests/conftest.py imports the heavy runtime)",
)

from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.schemas.callbacks_config import CallbackInstance
from synth_setter.schemas.datamodule_config import DataModuleConfig
from synth_setter.schemas.extras_config import ExtrasConfig
from synth_setter.schemas.logger_config import LoggerInstance
from synth_setter.schemas.model_config import ModelConfig, OptimizerConfig, SchedulerConfig
from synth_setter.schemas.paths_config import PathsConfig
from synth_setter.schemas.train_config import TrainConfig
from synth_setter.schemas.trainer_config import TrainerConfig

# mkdocs subprocess + griffe walk is heavyweight; keeps `make test-fast` fast.
pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RootModel wrappers (CallbacksConfig, LoggerConfig) expose only `root`; we
# document the per-instance class on those pages instead.
_PAGE_TO_MODELS: dict[str, tuple[type, ...]] = {
    "config_reference/dataset_spec": (DatasetSpec,),
    "config_reference/train_config": (TrainConfig,),
    "config_reference/datamodule_config": (DataModuleConfig,),
    "config_reference/model_config": (ModelConfig, OptimizerConfig, SchedulerConfig),
    "config_reference/callbacks_config": (CallbackInstance,),
    "config_reference/logger_config": (LoggerInstance,),
    "config_reference/trainer_config": (TrainerConfig,),
    "config_reference/paths_config": (PathsConfig,),
    "config_reference/extras_config": (ExtrasConfig,),
}


def _expected_field_anchors() -> list[tuple[str, type, str]]:
    """Build ``(page, model, field_name)`` tuples for every typed field on every model.

    :return: List of ``(page, model, field_name)`` triples for parametrization.
    """
    triples: list[tuple[str, type, str]] = []
    for page, models in _PAGE_TO_MODELS.items():
        for model in models:
            for field_name in model.model_fields:
                triples.append((page, model, field_name))
    return triples


assert _PAGE_TO_MODELS, "_PAGE_TO_MODELS is empty — config-reference page map is missing"
assert _expected_field_anchors(), "_expected_field_anchors() is empty — no fields to verify"


def _anchor_pattern(model: type, field_name: str) -> re.Pattern[str]:
    """Compile a regex for the ``id="<module>.<Class>.<field>"`` anchor mkdocstrings emits.

    Matches the structural anchor (not a bare substring) so short field names
    (``lr``, ``test``, ``train``) don't false-positive on HTML/CSS/JS context.

    :param model: Pydantic model class hosting the field.
    :param field_name: Name of the field being documented.
    :return: Compiled regex matching the expected mkdocstrings anchor id.
    """
    fqn = re.escape(f"{model.__module__}.{model.__name__}.{field_name}")
    return re.compile(rf'id="{fqn}"')


@pytest.fixture(scope="session")
def built_site(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run ``mkdocs build --strict`` once per session and return the site dir.

    :param tmp_path_factory: Pytest session-scoped tmp-path factory.
    :return: Path to the built mkdocs site directory.
    """
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


def _read_page_html(built_site: Path, page: str) -> str:
    """Read ``<built_site>/<page>/index.html`` with an actionable miss message.

    Pre-asserts ``is_file()`` so a missing page surfaces as a readable
    assertion instead of a bare ``FileNotFoundError`` from ``read_text``.

    :param built_site: Root directory of the built mkdocs site.
    :param page: Page subpath under the built site (no trailing slash).
    :return: HTML text of the page's ``index.html``.
    """
    page_html = built_site / page / "index.html"
    assert page_html.is_file(), (
        f"{page_html} missing from built site — mkdocs did not emit "
        f"{page} (see test_docs_page_emitted for the per-page check)"
    )
    return page_html.read_text()


@pytest.fixture(scope="session")
def page_html_cache(built_site: Path) -> dict[str, str]:
    """Memoised HTML text for each config-reference page.

    :param built_site: Session-scoped built mkdocs site directory.
    :return: Mapping ``page -> page_html`` for every documented page.
    """
    return {page: _read_page_html(built_site, page) for page in _PAGE_TO_MODELS}


@pytest.mark.parametrize("page", list(_PAGE_TO_MODELS))
def test_docs_page_emitted(built_site: Path, page: str) -> None:
    """Each config-reference page exists in the built site.

    :param built_site: Session-scoped built mkdocs site directory.
    :param page: Parametrized page subpath under the built site.
    """
    page_html = built_site / page / "index.html"
    assert page_html.is_file(), f"{page_html} missing from built site"


@pytest.mark.parametrize(
    ("page", "model", "field"),
    _expected_field_anchors(),
    ids=lambda v: v if isinstance(v, str) else getattr(v, "__name__", repr(v)),
)
def test_docs_page_renders_pydantic_field_anchor(
    page_html_cache: dict[str, str], page: str, model: type, field: str
) -> None:
    """Every typed field on every documented model gets its own anchor heading.

    :param page_html_cache: Session-cached mapping of page -> rendered HTML.
    :param page: Parametrized page subpath.
    :param model: Parametrized pydantic model class.
    :param field: Parametrized field name on ``model``.
    """
    assert page in page_html_cache, f"{page} missing from built site — see test_docs_page_emitted"
    page_html = page_html_cache[page]
    pattern = _anchor_pattern(model, field)
    assert pattern.search(page_html), (
        f"anchor {pattern.pattern!r} missing from {page} — "
        f"{model.__name__}.{field} pydantic field and docs page have drifted"
    )


def test_docs_train_config_renders_field_description(
    page_html_cache: dict[str, str],
) -> None:
    """``TrainConfig.task_name``'s ``Field(description=...)`` text renders in the page.

    :param page_html_cache: Session-cached mapping of page -> rendered HTML.
    """
    description = TrainConfig.model_fields["task_name"].description
    assert description, "TrainConfig.task_name has no Field(description=...) to spot-check"
    page_html = page_html_cache["config_reference/train_config"]
    assert description in page_html, (
        f"TrainConfig.task_name description {description!r} missing from rendered page — "
        f"mkdocstrings emitted the anchor but dropped the Field(description=...) body"
    )
