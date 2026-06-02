"""CI marker filters live only in the `make test-ci-*` targets — see #1353.

Finding 1 of the testing audit: every CI workflow re-spelled its pytest marker
expression inline, so they drifted (test.yml dropped `not requires_vst`,
nightly.yml silently ran `slow`). The marker strings now live in three Makefile
targets and the workflows call them. These tests pin both halves: the targets
carry the canonical expressions, and the workflows invoke the targets instead
of an inline `pytest -m`, so a future edit can't reintroduce drift unnoticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# make target -> the canonical marker expression its recipe must carry.
TARGET_MARKERS: dict[str, str] = {
    "test-ci-unit": "not slow and not gpu and not mps",
    "test-ci-slow": "slow and not gpu and not mps and not requires_vst",
    "test-ci-nightly": "not gpu and not mps and not requires_vst",
}

# workflow file -> the make target it must invoke instead of inline pytest.
WORKFLOW_TARGETS: dict[str, str] = {
    "test.yml": "test-ci-unit",
    "cpu-slow.yml": "test-ci-slow",
    "nightly.yml": "test-ci-nightly",
}


def _recipe(makefile: str, target: str) -> str:
    """Return the tab-indented recipe body for ``target`` in the Makefile text.

    :param makefile: full Makefile contents.
    :param target: target name whose recipe to extract.
    :returns: the recipe lines joined by newlines (empty if the target has none).
    """
    lines = makefile.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"{target}:"):
            recipe = []
            for body in lines[i + 1 :]:
                if body.startswith("\t"):
                    recipe.append(body)
                elif body.strip() == "":
                    continue
                else:
                    break
            return "\n".join(recipe)
    pytest.fail(f"Makefile missing target {target!r}")


@pytest.mark.infra
@pytest.mark.parametrize(("target", "marker_expr"), sorted(TARGET_MARKERS.items()))
def test_make_target_carries_canonical_marker(
    project_root: Path, target: str, marker_expr: str
) -> None:
    """Each `test-ci-*` recipe runs pytest with its canonical `-m` expression.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param target: the make target under test.
    :param marker_expr: the marker expression its recipe must pass to pytest.
    """
    recipe = _recipe((project_root / "Makefile").read_text(), target)
    assert f'-m "{marker_expr}"' in recipe, (
        f'`make {target}` recipe must run `pytest -m "{marker_expr}"`; got:\n{recipe}'
    )


@pytest.mark.infra
@pytest.mark.parametrize(("workflow", "target"), sorted(WORKFLOW_TARGETS.items()))
def test_workflow_invokes_make_target(project_root: Path, workflow: str, target: str) -> None:
    """The workflow calls `make <target>` rather than re-spelling pytest markers.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param workflow: workflow filename whose test step must call the target.
    :param target: the make target it must invoke.
    """
    text = (project_root / ".github" / "workflows" / workflow).read_text()
    invokes = re.search(rf"make {re.escape(target)}(?=\s|$)", text, re.MULTILINE)
    assert invokes is not None, (
        f"{workflow} must run `make {target}` so its marker filter stays in the "
        f"Makefile (see #1353)"
    )


@pytest.mark.infra
@pytest.mark.parametrize("workflow", sorted(WORKFLOW_TARGETS))
def test_workflow_has_no_inline_pytest_marker(project_root: Path, workflow: str) -> None:
    """No `pytest -m` survives inline — the make target is the only source.

    :param project_root: session fixture from ``tests/infra/conftest.py``.
    :param workflow: workflow filename that must not re-spell a marker filter.
    """
    text = (project_root / ".github" / "workflows" / workflow).read_text()
    # Collapse `\`-continuations so a `pytest \<newline>-m gpu` invocation split
    # across lines (the test-gpu.yml / test-mps.yml style) can't evade the guard.
    collapsed = text.replace("\\\n", " ")
    inline = re.search(r"pytest\b[^\n]*\s-m\s", collapsed)
    assert inline is None, (
        f"{workflow} re-spells a pytest marker filter inline ({inline.group(0)!r}); "
        f"move it into a `make test-ci-*` target to keep one source of truth (#1353)"
    )
