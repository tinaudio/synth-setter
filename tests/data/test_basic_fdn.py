"""Real offline/FLAMO parity for the canonical basic FDN boundary."""

import os
import selectors
import subprocess
import sys
import time
from dataclasses import replace
from textwrap import dedent
from typing import cast

import numpy as np
import pytest
import torch
from pyFDN import FDNBuild

from synth_setter.data.basic_fdn import BasicFDN

_TORCHSYNTH_BEHAVIOR_TIMEOUT_SECONDS = 30
_TORCHSYNTH_PROBE_PROGRESS = "torchsynth-probe-progress"
_TORCHSYNTH_PROBE_READY = "torchsynth-probe-ready"
_TORCHSYNTH_STARTUP_STALL_TIMEOUT_SECONDS = 30
_TORCHSYNTH_XDIST_GROUP = "torchsynth-compatibility-probe"


def _wait_for_torchsynth_probe_ready(
    process: subprocess.Popen[bytes], command: list[str]
) -> None:
    """Wait while bounded import stages report progress.

    :param process: Running fresh-process probe.
    :param command: Probe command for raised subprocess errors.
    :raises subprocess.CalledProcessError: If the child exits before readiness.
    :raises subprocess.TimeoutExpired: If one startup stage stops progressing.
    """
    assert process.stdout is not None
    buffered = bytearray()
    deadline = time.monotonic() + _TORCHSYNTH_STARTUP_STALL_TIMEOUT_SECONDS
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            newline_index = buffered.find(b"\n")
            if newline_index >= 0:
                status = bytes(buffered[:newline_index]).decode(errors="replace").strip()
                del buffered[: newline_index + 1]
                if status == _TORCHSYNTH_PROBE_READY:
                    return
                if status == _TORCHSYNTH_PROBE_PROGRESS:
                    deadline = time.monotonic() + _TORCHSYNTH_STARTUP_STALL_TIMEOUT_SECONDS
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(
                    command, _TORCHSYNTH_STARTUP_STALL_TIMEOUT_SECONDS
                )
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                raise subprocess.CalledProcessError(process.wait(), command)
            buffered.extend(chunk)


def _run_torchsynth_probe(probe: str) -> None:
    """Bound startup stalls and probe behavior separately.

    :param probe: Fresh-process Python source that reports import progress and readiness.
    :raises subprocess.CalledProcessError: If startup or probe behavior fails.
    :raises subprocess.TimeoutExpired: If startup stops progressing or probe behavior hangs.
    """
    command = [sys.executable, "-c", dedent(probe)]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        _wait_for_torchsynth_probe_ready(process, command)
        try:
            process.wait(timeout=_TORCHSYNTH_BEHAVIOR_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


@pytest.mark.xdist_group(name=_TORCHSYNTH_XDIST_GROUP)
def test_torchsynth_loader_preserves_canonical_pi_in_fresh_process() -> None:
    """The compatibility boundary repairs TorchSynth's process-global constant mutation."""
    probe = f"""
        import math
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        import torch
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        import torchsynth.util
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        from synth_setter.data.torchsynth_datamodule import _torchsynth_types

        print({_TORCHSYNTH_PROBE_READY!r}, flush=True)
        assert torch.pi != math.pi
        _torchsynth_types()
        assert torch.pi == math.pi
    """

    _run_torchsynth_probe(probe)


@pytest.mark.xdist_group(name=_TORCHSYNTH_XDIST_GROUP)
def test_torchsynth_loader_import_failure_preserves_canonical_pi_in_fresh_process() -> None:
    """A failed compatibility import still repairs TorchSynth's constant mutation."""
    probe = f"""
        import builtins
        import math
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        import torch
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        import torchsynth.util
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        from synth_setter.data.torchsynth_datamodule import _torchsynth_types

        print({_TORCHSYNTH_PROBE_READY!r}, flush=True)
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

    _run_torchsynth_probe(probe)


@pytest.mark.xdist_group(name=_TORCHSYNTH_XDIST_GROUP)
def test_torchsynth_loader_preserves_flamo_float64_parity_in_fresh_process() -> None:
    """A TorchSynth-first import leaves the real FLAMO response on its float64 baseline."""
    probe = f"""
        import numpy as np
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        import torch
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        import torchsynth.util
        print({_TORCHSYNTH_PROBE_PROGRESS!r}, flush=True)
        from pyFDN import FDNBuild
        from synth_setter.data.basic_fdn import BasicFDN
        from synth_setter.data.torchsynth_datamodule import _torchsynth_types

        print({_TORCHSYNTH_PROBE_READY!r}, flush=True)
        _torchsynth_types()
        build = FDNBuild(
            A=np.array([[0.2, 0.1], [-0.1, 0.2]]),
            B=np.array([[1.0, 0.3], [0.2, 1.0]]),
            C=np.array([[1.0, 0.4], [0.1, 1.0], [0.5, -0.2]]),
            D=np.array([[0.1, 0.0], [0.0, 0.2], [0.1, -0.1]]),
            delays=np.array([17, 29]),
            fs=48_000.0,
        )
        fdn = BasicFDN(build)
        expected = fdn.impulse_response(512)
        model = fdn.to_flamo(nfft=4096, device="cpu", dtype=torch.float64)
        impulse = torch.zeros(2, 4096, 2, dtype=torch.float64)
        impulse[:, 0, :] = torch.eye(2, dtype=torch.float64)
        actual = model(impulse).detach().numpy().transpose(1, 2, 0)[:512]

        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-7)
    """

    _run_torchsynth_probe(probe)


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
def test_basic_fdn_same_build_offline_and_flamo_responses_agree(
    build: FDNBuild, with_hooks: bool
) -> None:
    """Both real consumers preserve MIMO gains, delays and optional filter placement.

    :param build: Nontrivial complete FDN.
    :param with_hooks: Whether this case includes the three filter hooks.
    """
    if not with_hooks:
        build = replace(build, post_delay=None, post_matrix=None, post_output=None)
    fdn = BasicFDN(build)
    offline = fdn.impulse_response(512)
    model = fdn.to_flamo(nfft=4096, device="cpu", dtype=torch.float64)
    impulse = torch.zeros(2, 4096, 2, dtype=torch.float64)
    impulse[:, 0, :] = torch.eye(2)

    online = model(impulse).detach().numpy().transpose(1, 2, 0)

    assert offline.shape == (512, 3, 2)
    assert np.any(offline)
    np.testing.assert_allclose(online[:512], offline, atol=1e-9, rtol=1e-7)


def test_basic_fdn_source_build_mutation_does_not_change_render(build: FDNBuild) -> None:
    """The canonical build is independent of the caller's mutable NumPy arrays.

    :param build: Source build whose input matrix will be changed.
    """
    fdn = BasicFDN(build)
    expected = fdn.impulse_response(128)
    build.B[:] = 0.0

    np.testing.assert_array_equal(fdn.impulse_response(128), expected)


@pytest.mark.parametrize("nfft", [0, -1])
def test_basic_fdn_nonpositive_fft_period_rejected(build: FDNBuild, nfft: int) -> None:
    """Invalid graph geometry fails before FLAMO allocation.

    :param build: Valid FDN.
    :param nfft: Invalid FFT period.
    """
    with pytest.raises(ValueError, match="positive"):
        BasicFDN(build).to_flamo(nfft=nfft)


def test_basic_fdn_nonbuild_input_rejected_at_construction() -> None:
    """A custom renderer cannot masquerade as the basic build contract."""
    with pytest.raises(TypeError, match="FDNBuild"):
        BasicFDN(cast(FDNBuild, object()))


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
def test_basic_fdn_invalid_build_rejected_at_construction(
    build: FDNBuild, changes: dict, message: str
) -> None:
    """Malformed matrices or hooks fail before a training graph can be attached.

    :param build: Otherwise valid FDN.
    :param changes: Invalid replacement fields.
    :param message: Diagnostic identifying the violated contract.
    """
    with pytest.raises(ValueError, match=message):
        BasicFDN(replace(build, **changes))


@pytest.mark.parametrize("hook_name", ["post_delay", "post_matrix", "post_output"])
def test_basic_fdn_nonnormalized_filter_hook_rejected(
    build: FDNBuild, hook_name: str
) -> None:
    """Every optional filter hook requires a unit denominator coefficient.

    :param build: Otherwise valid FDN.
    :param hook_name: Filter hook made nonnormalized.
    """
    hook = np.array(getattr(build, hook_name), copy=True)
    hook[:, 3, :] = 2.0

    with pytest.raises(ValueError, match="normalized"):
        BasicFDN(replace(build, **{hook_name: hook}))


@pytest.mark.parametrize("length", [0, -1])
def test_basic_fdn_nonpositive_render_length_rejected(build: FDNBuild, length: int) -> None:
    """Invalid render geometry fails before upstream allocation.

    :param build: Valid FDN.
    :param length: Invalid number of output samples.
    """
    with pytest.raises(ValueError, match="positive"):
        BasicFDN(build).impulse_response(length)
