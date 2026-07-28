"""Behavior tests for the modeling typing lint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINTER = PROJECT_ROOT / "scripts" / "lint_model_typing.py"


def _run_linter(models_dir: Path, baseline: Path) -> subprocess.CompletedProcess[str]:
    """Run the real lint CLI against an isolated package.

    :param models_dir: Modeling package fixture.
    :param baseline: Baseline fixture for the package.
    :returns: Completed lint process with captured output.
    """
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository script.
        [
            sys.executable,
            str(LINTER),
            "--models-dir",
            str(models_dir),
            "--baseline",
            str(baseline),
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_linter_compliant_jaxtyping_callable_passes(tmp_path: Path) -> None:
    """Accept a shape-annotated callable with runtime checking.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "encoder.py").write_text(
        """from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

@jaxtyped(typechecker=beartype)
def encode(samples: Float[Tensor, \"batch samples\"]) -> Float[Tensor, \"batch features\"]:
    return samples
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 0, result.stdout + result.stderr


def test_linter_new_modeling_callable_reports_missing_runtime_check(tmp_path: Path) -> None:
    """Reject a callable without the beartype-backed decorator.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "encoder.py").write_text(
        """from jaxtyping import Float
from torch import Tensor

def encode(samples: Float[Tensor, \"batch samples\"]):
    return samples
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "encoder.py:encode:JAX001" in result.stdout
    assert "@jaxtyped(typechecker=beartype)" in result.stdout


def test_linter_bare_torch_tensor_reports_shape_annotation_and_runtime_check(
    tmp_path: Path,
) -> None:
    """Reject bare tensor annotations and missing runtime checking.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "decoder.py").write_text(
        """import torch

def decode(latent: torch.Tensor) -> torch.Tensor:
    return latent
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "decoder.py:decode:JAX001" in result.stdout
    assert "decoder.py:decode:JAX002" in result.stdout
    assert "jaxtyping annotation" in result.stdout


def test_linter_non_jaxtyping_generic_does_not_hide_bare_tensor(tmp_path: Path) -> None:
    """Reject a bare tensor nested in an unrelated generic.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "wrapped.py").write_text(
        """import torch

class Wrapper: ...

def encode(value: Wrapper[torch.Tensor]):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "wrapped.py:encode:JAX002" in result.stdout


def test_linter_baseline_grandfathers_matching_violations(tmp_path: Path) -> None:
    """Accept violations that exactly match the legacy baseline.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "legacy.py").write_text(
        """import torch

def legacy_forward(value: torch.Tensor) -> torch.Tensor:
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("legacy.py:legacy_forward:JAX001\nlegacy.py:legacy_forward:JAX002\n")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 0, result.stdout + result.stderr


def test_precommit_runs_model_typing_lint_for_models_and_baseline() -> None:
    """Register the linter for every input that can affect its result."""
    config = (PROJECT_ROOT / ".pre-commit-config.yaml").read_text()

    assert "id: model-typing" in config
    assert "entry: uv run python scripts/lint_model_typing.py" in config
    assert ".model-typing-baseline\\.txt" in config
    assert "src/synth_setter/models/" in config


def test_linter_stale_baseline_entry_fails(tmp_path: Path) -> None:
    """Require removal of baseline entries after violations are fixed.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "typed.py").write_text(
        """from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

@jaxtyped(typechecker=beartype)
def typed(value: Float[Tensor, \"batch\"]):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("typed.py:typed:JAX001\n")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "stale baseline entry: typed.py:typed:JAX001" in result.stdout
