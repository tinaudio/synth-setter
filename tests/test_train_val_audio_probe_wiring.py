"""Tests for ``_configure_val_audio_probe``'s opt-in gating and URI derivation.

``ensure_r2_env_loaded`` is stubbed out throughout: the probe calls it to fail fast
on absent R2 credentials, but it pings the live remote, which these tests neither
have nor need. The upload itself is exercised against a real rclone in
``test_train.py::test_train_surge_xt_val_audio_probe``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lightning import Callback
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import _configure_val_audio_probe, _derive_probe_uri
from synth_setter.pipeline import r2_io
from synth_setter.utils.callbacks import ValAudioProbe


@pytest.fixture(autouse=True)
def _skip_r2_auth_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the R2 pre-flight so these tests need no credentials.

    :param monkeypatch: Replaces the auth ping with a no-op.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)


def _cfg(*, enabled: bool, with_render: bool = True, output_dir: str = "/runs/out") -> DictConfig:
    """Build the minimal train cfg slice ``_configure_val_audio_probe`` reads.

    :param enabled: Value for ``training.val_audio_probe``.
    :param with_render: When ``False``, omit the ``render`` group entirely.
    :param output_dir: Value for ``paths.output_dir``.
    :returns: Composed cfg fragment.
    """
    cfg = OmegaConf.create(
        {
            "task_name": "train",
            "r2": {"bucket": "intermediate-data"},
            "paths": {"output_dir": output_dir},
            "training": {"val_audio_probe": enabled, "val_audio_probe_samples": 5},
        }
    )
    if with_render:
        with open_dict(cfg):
            cfg.render = {
                "param_spec_name": "surge_xt",
                "plugin_state_path": "presets/surge-base.vstpreset",
                "plugin_path": "plugins/Surge XT.vst3",
                "sample_rate": 44100,
                "channels": 2,
                "velocity": 100,
                "signal_duration_seconds": 4.0,
            }
    return cfg


def test_configure_val_audio_probe_appends_nothing_when_disabled() -> None:
    """The default-off flag leaves the callback list untouched."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=False), callbacks)

    assert callbacks == []


def test_configure_val_audio_probe_appends_probe_when_enabled() -> None:
    """Enabling the flag wires exactly one ValAudioProbe under the run's output dir."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=True), callbacks)

    assert len(callbacks) == 1
    probe = callbacks[0]
    assert isinstance(probe, ValAudioProbe)
    assert probe.num_samples == 5
    assert probe.probe_root == Path("/runs/out") / "val_audio_probe"


def test_configure_val_audio_probe_raises_when_render_group_missing() -> None:
    """Enabling the probe without a render group fails with a directed error."""
    with pytest.raises(ValueError, match="render"):
        _configure_val_audio_probe(_cfg(enabled=True, with_render=False), [])


def test_derive_probe_uri_uses_bucket_and_run_config_id() -> None:
    """The snapshot prefix derives from r2.bucket under probes/."""
    uri = _derive_probe_uri(_cfg(enabled=True))

    assert uri.startswith("r2://intermediate-data/probes/")
