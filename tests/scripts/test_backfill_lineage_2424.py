"""Unit tests for the #2424 lineage-backfill planner.

The planner is the half of the backfill that decides *what* to repair, so it is
pure and asserted directly on its return values; the apply half is a thin
sequence of ``wandb`` API calls over that plan. Both planners must be idempotent
— the script is expected to be re-run after a partial failure — so every
"already repaired" case is pinned alongside its "needs repair" counterpart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dev/backfill_lineage_2424.py"


def _load_script() -> ModuleType:
    """Import the backfill script by path — ``scripts/dev`` is not an importable package.

    :returns: The imported module.
    """
    spec = importlib.util.spec_from_file_location("backfill_lineage_2424", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backfill = _load_script()

_PRODUCER = "surge-simple-lance-440k-20k-20k-20260706T005448315Z"
_DATASET = "data-surge-simple-lance-440k-20k-20k"


class TestPlanAlias:
    """Deciding whether the artifact still needs its immutable producing-run alias."""

    def test_absent_producing_run_alias_is_planned(self) -> None:
        """A pre-#1881 artifact carrying only ``latest`` needs the run-id alias added."""
        assert backfill.plan_alias(["latest"], _PRODUCER) == _PRODUCER

    def test_present_producing_run_alias_plans_nothing(self) -> None:
        """Re-running after a successful backfill adds nothing (the script is idempotent)."""
        assert backfill.plan_alias(["latest", _PRODUCER], _PRODUCER) is None

    def test_unknown_producing_run_plans_nothing(self) -> None:
        """With no producing run there is no alias to derive, so the repair is skipped."""
        assert backfill.plan_alias(["latest"], None) is None


class TestPlanEdges:
    """Deciding which consuming runs still lack the dataset ``use_artifact`` edge."""

    def test_run_with_no_used_artifacts_is_planned(self) -> None:
        """The #2424 symptom — ``used_artifacts() == []`` — plans the run for repair."""
        consumed = {"run-a": []}

        assert backfill.plan_edges(consumed, _DATASET) == ("run-a",)

    def test_run_already_consuming_the_dataset_is_skipped(self) -> None:
        """A run whose edge already exists is left alone, whatever version it names."""
        consumed = {"run-a": [f"{_DATASET}:v0"]}

        assert backfill.plan_edges(consumed, _DATASET) == ()

    def test_run_consuming_only_other_artifacts_is_planned(self) -> None:
        """An unrelated input edge does not count as the dataset edge."""
        consumed = {"run-a": ["model-flow-simple:latest"]}

        assert backfill.plan_edges(consumed, _DATASET) == ("run-a",)

    def test_planned_runs_preserve_input_order(self) -> None:
        """Runs are repaired in the order given so the dry-run report is stable."""
        consumed = {"run-b": [], "run-a": [], "run-c": [f"{_DATASET}:v0"]}

        assert backfill.plan_edges(consumed, _DATASET) == ("run-b", "run-a")
