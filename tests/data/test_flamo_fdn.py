"""Real offline/FLAMO parity for the canonical FLAMO FDN boundary."""

import subprocess
import sys
from dataclasses import replace
from textwrap import dedent
from typing import cast

import numpy as np
import pytest
import torch
from pyFDN import FDNBuild

from synth_setter.data.flamo_fdn import FlamoFDN

_TORCHSYNTH_IMPORT_TIMEOUT_SECONDS = 30


def test_torchsynth_loader_preserves_canonical_pi_in_fresh_process() -> None:
    """The compatibility boundary repairs TorchSynth's process-global constant mutation."""
    probe = """
        import math
        import torch
        import torchsynth.util
        from synth_setter.data.torchsynth_datamodule import _torchsynth_types

        assert torch.pi != math.pi
        _torchsynth_types()
        assert torch.pi == math.pi
    """

    subprocess.run(
        [sys.executable, "-c", dedent(probe)],
        check=True,
        timeout=_TORCHSYNTH_IMPORT_TIMEOUT_SECONDS,
    )


def test_torchsynth_loader_import_failure_preserves_canonical_pi_in_fresh_process() -> None:
    """A failed compatibility import still repairs TorchSynth's constant mutation."""
    probe = """
        import builtins
        import math
        import torch
        import torchsynth.util
        from synth_setter.data.torchsynth_datamodule import _torchsynth_types

        original_import = builtins.__import__
        def fail_synth_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "torchsynth.synth":
                raise ImportError("expected probe failure")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = fail_synth_import
        try:
            _torchsynth_types()
        except ImportError:
            pass
        else:
            raise AssertionError("TorchSynth import unexpectedly succeeded")
        assert torch.pi == math.pi
    """

    subprocess.run(
        [sys.executable, "-c", dedent(probe)],
        check=True,
        timeout=_TORCHSYNTH_IMPORT_TIMEOUT_SECONDS,
    )


def test_torchsynth_loader_preserves_flamo_float64_parity_in_fresh_process() -> None:
    """A TorchSynth-first import leaves the real FLAMO response on its float64 baseline."""
    probe = """
        import numpy as np
        import torch
        import torchsynth.util
        from pyFDN import FDNBuild
        from synth_setter.data.flamo_fdn import FlamoFDN
        from synth_setter.data.torchsynth_datamodule import _torchsynth_types

        _torchsynth_types()
        build = FDNBuild(
            A=np.array([[0.2, 0.1], [-0.1, 0.2]]),
            B=np.array([[1.0, 0.3], [0.2, 1.0]]),
            C=np.array([[1.0, 0.4], [0.1, 1.0], [0.5, -0.2]]),
            D=np.array([[0.1, 0.0], [0.0, 0.2], [0.1, -0.1]]),
            delays=np.array([17, 29]),
            fs=48_000.0,
        )
        fdn = FlamoFDN(build)
        expected = fdn.impulse_response(512)
        model = fdn.to_flamo(nfft=4096, device="cpu", dtype=torch.float64)
        impulse = torch.zeros(2, 4096, 2, dtype=torch.float64)
        impulse[:, 0, :] = torch.eye(2, dtype=torch.float64)
        actual = model(impulse).detach().numpy().transpose(1, 2, 0)[:512]

        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-7)
    """

    subprocess.run(
        [sys.executable, "-c", dedent(probe)],
        check=True,
        timeout=_TORCHSYNTH_IMPORT_TIMEOUT_SECONDS,
    )


@pytest.fixture
def build() -> FDNBuild:
    """A non-Householder MIMO build with distinct filters at every hook.

    :returns: Complete stable build for the two real renderers.
    """
    return FDNBuild(
        A=np.array([[0.2, 0.1], [-0.1, 0.2]]),
        B=np.array([[1.0, 0.3], [0.2, 1.0]]),
        C=np.array([[1.0, 0.4], [0.1, 1.0], [0.5, -0.2]]),
        D=np.array([[0.1, 0.0], [0.0, 0.2], [0.1, -0.1]]),
        delays=np.array([17, 29]),
        fs=48_000.0,
        post_delay=np.array([[[0.8, 0.7], [0.1, 0.05], [0.0, 0.0],
                              [1.0, 1.0], [-0.1, -0.2], [0.0, 0.0]]]),
        post_matrix=np.array([[[0.6, 0.9], [0.05, 0.02], [0.0, 0.0],
                               [1.0, 1.0], [-0.15, -0.1], [0.0, 0.0]]]),
        post_output=np.array([[[1.1, 0.9, 0.8], [0.05, 0.1, 0.02], [0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0], [-0.1, -0.2, -0.15], [0.0, 0.0, 0.0]]]),
    )


@pytest.mark.parametrize("with_hooks", [False, True])
def test_flamo_fdn_same_build_offline_and_flamo_responses_agree(
    build: FDNBuild, with_hooks: bool
) -> None:
    """Both real consumers preserve MIMO gains, delays and optional filter placement.

    :param build: Nontrivial complete FDN.
    :param with_hooks: Whether this case includes the three filter hooks.
    """
    if not with_hooks:
        build = replace(build, post_delay=None, post_matrix=None, post_output=None)
    fdn = FlamoFDN(build)
    offline = fdn.impulse_response(512)
    model = fdn.to_flamo(nfft=4096, device="cpu", dtype=torch.float64)
    impulse = torch.zeros(2, 4096, 2, dtype=torch.float64)
    impulse[:, 0, :] = torch.eye(2)

    online = model(impulse).detach().numpy().transpose(1, 2, 0)

    assert offline.shape == (512, 3, 2)
    assert np.any(offline)
    np.testing.assert_allclose(online[:512], offline, atol=1e-9, rtol=1e-7)


def test_flamo_fdn_source_build_mutation_does_not_change_render(build: FDNBuild) -> None:
    """The canonical build is independent of the caller's mutable NumPy arrays.

    :param build: Source build whose input matrix will be changed.
    """
    fdn = FlamoFDN(build)
    expected = fdn.impulse_response(128)
    build.B[:] = 0.0

    np.testing.assert_array_equal(fdn.impulse_response(128), expected)


@pytest.mark.parametrize("nfft", [0, -1])
def test_flamo_fdn_nonpositive_fft_period_rejected(build: FDNBuild, nfft: int) -> None:
    """Invalid graph geometry fails before FLAMO allocation.

    :param build: Valid FDN.
    :param nfft: Invalid FFT period.
    """
    with pytest.raises(ValueError, match="positive"):
        FlamoFDN(build).to_flamo(nfft=nfft)


def test_flamo_fdn_nonbuild_input_rejected_at_construction() -> None:
    """A custom renderer cannot masquerade as the basic build contract."""
    with pytest.raises(TypeError, match="FDNBuild"):
        FlamoFDN(cast(FDNBuild, object()))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"A": np.ones((2, 3))}, "square"),
        ({"fs": float("nan")}, "sample_rate"),
        ({"delays": np.array([0, 2])}, "positive integer"),
        ({"post_output": np.ones((1, 6, 2))}, "post_output"),
        ({"post_delay": np.zeros((0, 6, 2))}, "post_delay"),
        ({"post_matrix": np.zeros((1, 6, 2))}, "normalized"),
    ],
)
def test_flamo_fdn_invalid_build_rejected_at_construction(
    build: FDNBuild, changes: dict, message: str
) -> None:
    """Malformed matrices or hooks fail before a training graph can be attached.

    :param build: Otherwise valid FDN.
    :param changes: Invalid replacement fields.
    :param message: Diagnostic identifying the violated contract.
    """
    with pytest.raises(ValueError, match=message):
        FlamoFDN(replace(build, **changes))


@pytest.mark.parametrize("hook_name", ["post_delay", "post_matrix", "post_output"])
def test_flamo_fdn_nonnormalized_filter_hook_rejected(
    build: FDNBuild, hook_name: str
) -> None:
    """Every optional filter hook requires a unit denominator coefficient.

    :param build: Otherwise valid FDN.
    :param hook_name: Filter hook made nonnormalized.
    """
    hook = np.array(getattr(build, hook_name), copy=True)
    hook[:, 3, :] = 2.0

    with pytest.raises(ValueError, match="normalized"):
        FlamoFDN(replace(build, **{hook_name: hook}))


@pytest.mark.parametrize("length", [0, -1])
def test_flamo_fdn_nonpositive_render_length_rejected(build: FDNBuild, length: int) -> None:
    """Invalid render geometry fails before upstream allocation.

    :param build: Valid FDN.
    :param length: Invalid number of output samples.
    """
    with pytest.raises(ValueError, match="positive"):
        FlamoFDN(build).impulse_response(length)
