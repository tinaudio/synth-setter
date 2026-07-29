"""Behavior tests for the modeling typing lint."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINTER = PROJECT_ROOT / "scripts" / "lint_model_typing.py"
GIT = shutil.which("git") or "git"
UV = shutil.which("uv") or "uv"


def _run_linter(
    models_dir: Path,
    baseline: Path,
    *,
    base_ref: str | None = None,
    allow_missing_git_base: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run the real lint CLI against an isolated package.

    :param models_dir: Modeling package fixture.
    :param baseline: Baseline fixture for the package.
    :param base_ref: Optional git ref that owns the frozen baseline.
    :param allow_missing_git_base: Permit non-git unit-test fixtures.
    :returns: Completed lint process with captured output.
    """
    env = os.environ.copy()
    if base_ref is not None:
        env["MODEL_TYPING_BASE_REF"] = base_ref
    command = [
        sys.executable,
        str(LINTER),
        "--models-dir",
        str(models_dir),
        "--baseline",
        str(baseline),
    ]
    if allow_missing_git_base:
        command.append("--allow-missing-git-base")
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository script.
        command,
        capture_output=True,
        check=False,
        cwd=baseline.parent,
        env=env,
        text=True,
    )


def _commit_all(repo: Path) -> None:
    """Initialize a test repository and commit its current files.

    :param repo: Temporary repository root.
    """
    subprocess.run([GIT, "init", "-q"], cwd=repo, check=True)  # noqa: S603
    subprocess.run([GIT, "add", "."], cwd=repo, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [
            GIT,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=repo,
        check=True,
    )


def _configured_model_hook() -> dict[str, Any]:
    """Return the model-typing hook from the repository configuration.

    :returns: Parsed local hook configuration.
    :raises AssertionError: If the hook is absent.
    """
    config = yaml.safe_load((PROJECT_ROOT / ".pre-commit-config.yaml").read_text())
    for repo in config["repos"]:
        if repo["repo"] != "local":
            continue
        for hook in repo["hooks"]:
            if hook["id"] == "model-typing":
                return hook
    raise AssertionError("model-typing hook is not configured")


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


def test_linter_qualified_runtime_decorator_passes(tmp_path: Path) -> None:
    """Accept qualified jaxtyping and beartype decorator references.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "encoder.py").write_text(
        """import beartype
import jaxtyping
from torch import Tensor

@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def encode(samples: jaxtyping.Float[Tensor, \"batch samples\"]):
    return samples
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 0, result.stdout + result.stderr


def test_linter_nested_fake_imports_do_not_satisfy_runtime_check(tmp_path: Path) -> None:
    """Ignore imports that do not establish module-level decorator names.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "nested.py").write_text(
        """def configure():
    from beartype import beartype
    from jaxtyping import jaxtyped

@jaxtyped(typechecker=beartype)
def encode(value):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "nested.py:encode:JAX001" in result.stdout


def test_linter_rebound_imports_do_not_satisfy_runtime_check(tmp_path: Path) -> None:
    """Reject decorator names rebound after canonical imports.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "rebound.py").write_text(
        """from beartype import beartype
from jaxtyping import jaxtyped

def jaxtyped(*args, **kwargs):
    return lambda function: function

@jaxtyped(typechecker=beartype)
def encode(value):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "rebound.py:encode:JAX001" in result.stdout


def test_linter_later_fake_import_invalidates_jaxtyped_binding(tmp_path: Path) -> None:
    """Resolve decorator imports in module source order.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "overwritten.py").write_text(
        """from beartype import beartype
from jaxtyping import jaxtyped
from project.fake_typing import jaxtyped

@jaxtyped(typechecker=beartype)
def encode(value):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "overwritten.py:encode:JAX001" in result.stdout


def test_linter_relative_typing_imports_do_not_satisfy_runtime_check(tmp_path: Path) -> None:
    """Reject local modules masquerading as typing dependencies.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "relative.py").write_text(
        """from .beartype import beartype
from .jaxtyping import jaxtyped

@jaxtyped(typechecker=beartype)
def encode(value):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "relative.py:encode:JAX001" in result.stdout


def test_linter_compound_rebinding_invalidates_jaxtyped_import(tmp_path: Path) -> None:
    """Reject canonical decorator names overwritten in compound statements.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "compound.py").write_text(
        """from beartype import beartype
from jaxtyping import jaxtyped

if True:
    jaxtyped = lambda **kwargs: lambda function: function

@jaxtyped(typechecker=beartype)
def encode(value):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "compound.py:encode:JAX001" in result.stdout


def test_linter_comprehension_target_does_not_hide_torch_import(tmp_path: Path) -> None:
    """Keep comprehension-local targets out of module binding resolution.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "comprehension.py").write_text(
        """import torch

items = [torch for torch in ()]

def encode(value: torch.Tensor):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "comprehension.py:encode:JAX002" in result.stdout


def test_linter_class_scope_typing_imports_pass(tmp_path: Path) -> None:
    """Accept canonical imports active in a method's class scope.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "scoped_class.py").write_text(
        """class Encoder:
    from beartype import beartype
    from jaxtyping import Float, jaxtyped
    from torch import Tensor

    @jaxtyped(typechecker=beartype)
    def encode(self, value: Float[Tensor, \"features\"]):
        return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "scoped_class.py:Encoder.encode:JAX001" not in result.stdout
    assert "scoped_class.py:Encoder.encode:JAX002" not in result.stdout
    assert "scoped_class.py:Encoder:JAX001" not in result.stdout


def test_linter_enclosing_function_typing_imports_pass(tmp_path: Path) -> None:
    """Accept typing imports active in an enclosing function scope.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "nested_valid.py").write_text(
        """from beartype import beartype
from jaxtyping import jaxtyped

@jaxtyped(typechecker=beartype)
def outer():
    from beartype import beartype as bt
    from jaxtyping import jaxtyped as jt

    @jt(typechecker=bt)
    def inner(value):
        return value

    return inner
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


def test_linter_assigned_torch_tensor_alias_reports_shape_annotation(tmp_path: Path) -> None:
    """Reject aliases assigned from torch.Tensor.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "assigned.py").write_text(
        """import torch

TorchTensor = torch.Tensor

def encode(value: TorchTensor):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "assigned.py:encode:JAX002" in result.stdout


def test_linter_aliased_torch_tensor_reports_shape_annotation(tmp_path: Path) -> None:
    """Reject torch tensor aliases outside a jaxtyping wrapper.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "aliased.py").write_text(
        """import torch as th
from torch import Tensor as TorchTensor

def encode(value: TorchTensor) -> th.Tensor:
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "aliased.py:encode:JAX002" in result.stdout


def test_linter_unrelated_local_torch_parameter_preserves_module_binding(tmp_path: Path) -> None:
    """Keep module imports visible outside unrelated local scopes.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "scoped.py").write_text(
        """import torch

def helper(torch: str):
    return torch

def encode(value: torch.Tensor):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "scoped.py:encode:JAX002" in result.stdout


def test_linter_import_torch_submodule_still_resolves_torch_binding(tmp_path: Path) -> None:
    """Recognize the torch binding established by a submodule import.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "submodule.py").write_text(
        """import torch.nn

def encode(value: torch.Tensor):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "submodule.py:encode:JAX002" in result.stdout


def test_linter_quoted_torch_tensor_reports_shape_annotation(tmp_path: Path) -> None:
    """Reject bare tensors encoded as forward-reference strings.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "quoted.py").write_text(
        """import torch
from beartype import beartype
from jaxtyping import jaxtyped

@jaxtyped(typechecker=beartype)
def encode(value: \"torch.Tensor\"):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "quoted.py:encode:JAX002" in result.stdout


def test_linter_malformed_quoted_torch_tensor_reports_shape_annotation(tmp_path: Path) -> None:
    """Reject a tensor forward reference even when its expression is malformed.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "malformed.py").write_text(
        """import torch

def encode(value: \"torch.Tensor[\"):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "malformed.py:encode:JAX002" in result.stdout


def test_linter_malformed_forward_reference_respects_identifier_boundary(tmp_path: Path) -> None:
    """Do not mistake TensorLike for an imported Tensor name.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "tensor_like.py").write_text(
        """from torch import Tensor

def encode(value: \"TensorLike[\"):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "tensor_like.py:encode:JAX001" in result.stdout
    assert "tensor_like.py:encode:JAX002" not in result.stdout


def test_linter_starred_torch_tensor_reports_shape_annotation(tmp_path: Path) -> None:
    """Reject bare tensors nested beneath starred annotation syntax.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "starred.py").write_text(
        """import torch

def encode(value: tuple[*tuple[torch.Tensor, ...]]):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 1
    assert "starred.py:encode:JAX002" in result.stdout


def test_linter_imported_jaxtyping_wrapper_passes_without_allowlist(tmp_path: Path) -> None:
    """Accept any annotation wrapper imported from jaxtyping.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "float8.py").write_text(
        """from beartype import beartype
from jaxtyping import Float8e4m3fn, jaxtyped
from torch import Tensor

@jaxtyped(typechecker=beartype)
def encode(value: Float8e4m3fn[Tensor, \"batch\"]):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")

    result = _run_linter(models_dir, baseline)

    assert result.returncode == 0, result.stdout + result.stderr


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


def test_precommit_model_typing_hook_rejects_violating_model(tmp_path: Path) -> None:
    """Run the configured hook against a staged violating model.

    :param tmp_path: Isolated git repository for the real pre-commit invocation.
    """
    (tmp_path / ".pre-commit-config.yaml").write_text(
        yaml.safe_dump({"repos": [{"repo": "local", "hooks": [_configured_model_hook()]}]})
    )
    (tmp_path / "scripts").mkdir()
    shutil.copy2(LINTER, tmp_path / "scripts" / LINTER.name)
    (tmp_path / ".model-typing-baseline.txt").write_text("")
    _commit_all(tmp_path)
    model = tmp_path / "src" / "synth_setter" / "models" / "new.py"
    model.parent.mkdir(parents=True)
    model.write_text("def new(value):\n    return value\n")
    subprocess.run([GIT, "add", str(model)], cwd=tmp_path, check=True)  # noqa: S603
    env = os.environ.copy()
    env["MODEL_TYPING_BASE_REF"] = "HEAD"

    result = subprocess.run(  # noqa: S603
        [UV, "run", "pre-commit", "run", "model-typing", "--files", str(model)],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode == 1
    assert "new.py:new:JAX001" in result.stdout


def test_linter_baseline_addition_against_git_ref_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject additions to a baseline committed on the explicit base ref.

    :param tmp_path: Isolated lint fixture directory.
    :param monkeypatch: Environment isolation for the base-ref override.
    """
    monkeypatch.setenv("GITHUB_BASE_REF", "wrong-ci-base")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    _commit_all(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "new.py").write_text(
        """import torch

def new(value: torch.Tensor):
    return value
"""
    )
    baseline.write_text("new.py:new:JAX001\nnew.py:new:JAX002\n")

    result = _run_linter(
        models_dir,
        baseline,
        base_ref="HEAD",
        allow_missing_git_base=False,
    )

    assert result.returncode == 1
    assert "baseline additions are forbidden" in result.stdout
    assert "new.py:new:JAX001" in result.stdout
    assert "new.py:new:JAX002" in result.stdout


def test_linter_github_checkout_without_remote_uses_first_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the checkout commit's first parent when the base remote is absent.

    :param tmp_path: Isolated git repository.
    :param monkeypatch: GitHub Actions environment fixture.
    """
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    _commit_all(tmp_path)
    (tmp_path / "dummy.txt").write_text("merge checkout\n")
    subprocess.run([GIT, "add", "dummy.txt"], cwd=tmp_path, check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [
            GIT,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "merge checkout",
        ],
        cwd=tmp_path,
        check=True,
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "new.py").write_text("def new(value):\n    return value\n")
    baseline.write_text("new.py:new:JAX001\n")
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    monkeypatch.delenv("MODEL_TYPING_BASE_REF", raising=False)

    result = _run_linter(
        models_dir,
        baseline,
        allow_missing_git_base=False,
    )

    assert result.returncode == 1
    assert "baseline additions are forbidden: new.py:new:JAX001" in result.stdout


def test_linter_missing_git_base_fails_closed(tmp_path: Path) -> None:
    """Reject baseline validation outside a git repository by default.

    :param tmp_path: Isolated lint fixture directory.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "legacy.py").write_text(
        """def legacy(value):
    return value
"""
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("legacy.py:legacy:JAX001\n")

    result = _run_linter(models_dir, baseline, allow_missing_git_base=False)

    assert result.returncode == 1
    assert "cannot locate git root for model typing baseline" in result.stdout


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
