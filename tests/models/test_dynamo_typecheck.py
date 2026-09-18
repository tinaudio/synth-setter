"""Behavior tests for the jaxtyping/Dynamo type-check bypass."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from beartype import beartype
from jaxtyping import Float, TypeCheckError, jaxtyped
from jaxtyping import _config as jaxtyping_config
from torch import Tensor, nn

from synth_setter.models.components.slap import SiameseArm
from synth_setter.models.dynamo_typecheck import install_dynamo_typecheck_bypass

_MODELS_DIR = Path(__file__).resolve().parents[2] / "src/synth_setter/models"


class _Doubler(nn.Module):
    """Typed module standing in for any annotated model the bypass has to leave checked."""

    @jaxtyped(typechecker=beartype)
    def forward(self, audio: Float[Tensor, "batch samples"]) -> Float[Tensor, "batch samples"]:
        """Double the waveform.

        :param audio: Mono waveform batch.
        :returns: The input scaled by two.
        """
        return audio * 2.0


def test_explicit_disable_still_turns_runtime_checking_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator's ``JAXTYPING_DISABLE=1`` must keep silencing checks after installation.

    :param monkeypatch: Restores the global switch once the test returns.
    """
    install_dynamo_typecheck_bypass()
    monkeypatch.setattr(jaxtyping_config.config, "jaxtyping_disable", True)

    assert torch.equal(_Doubler()(torch.ones(2, 3, dtype=torch.int64)), torch.full((2, 3), 2))


def test_repeated_installation_keeps_eager_type_checking_on() -> None:
    """Installing twice must not stack wrappers or silence the eager check."""
    install_dynamo_typecheck_bypass()
    install_dynamo_typecheck_bypass()

    with pytest.raises(TypeCheckError):
        _Doubler()(torch.zeros(2, 3, dtype=torch.int64))


def test_compiled_siamese_arm_accepts_input_its_annotation_allows() -> None:
    """A real nested-typed component must compile and run, not just a local stand-in (#3225).

    `SiameseArm.forward` reaches further `jaxtyped` calls while tracing, which is the shape
    that desynchronizes jaxtyping's memo stack; a flat stand-in cannot exercise it.
    """
    install_dynamo_typecheck_bypass()
    arm = SiameseArm(nn.Linear(2, 4), nn.Linear(4, 3))

    representation, projection, prediction = torch.compile(arm, backend="eager")(torch.randn(2, 2))

    assert representation.shape == (2, 4)
    assert projection.shape == (2, 3)
    assert prediction.shape == (2, 3)


def test_importing_the_models_package_does_not_import_torch() -> None:
    """The package root stays torch-free so coverage can resolve a submodule source.

    `coverage run --source=synth_setter.models.<mod>` resolves that source by importing the
    parent package before conftest runs. Pulling torch in there re-enters `torch/__init__` and
    aborts the interpreter in `torch._C` — see #3572.
    """
    probe = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", "import synth_setter.models, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.strip() == "False", "synth_setter.models must not import torch at import"


def _compiling_setups() -> list[tuple[str, ast.FunctionDef]]:
    """Return every ``setup`` in the models package that compiles a submodule.

    :returns: Pairs of module filename and its ``setup`` definition.
    """
    found: list[tuple[str, ast.FunctionDef]] = []
    for path in sorted(Path(_MODELS_DIR).glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "setup":
                continue
            calls = {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            }
            if "compile" in calls:
                found.append((path.name, node))
    return found


def test_every_compiling_setup_installs_the_bypass() -> None:
    """A compile site that skips the install silently loses the #3572 fix.

    The bypass is installed per compile site rather than at package import, because importing
    torch from ``synth_setter/models/__init__.py`` aborts the interpreter under coverage.
    """
    setups = _compiling_setups()

    assert setups, "no compiling setup() found — the guard would pass vacuously"
    missing = [
        name
        for name, node in setups
        if "install_dynamo_typecheck_bypass"
        not in {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
    ]
    assert not missing, f"compile sites without the jaxtyping bypass: {missing}"
