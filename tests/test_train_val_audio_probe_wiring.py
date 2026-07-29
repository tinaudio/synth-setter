"""Tests for ``_configure_val_audio_probe`` mode gating and URI derivation.

``ensure_r2_env_loaded`` is stubbed out throughout: the probe calls it to fail fast
on absent R2 credentials, but it pings the live remote, which these tests neither
have nor need. The upload itself is exercised against a real rclone in
``test_train.py::test_train_surge_xt_val_audio_probe``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pytest
import torch
from lightning import Callback
from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import ValidationError

from synth_setter.cli.train import (
    _checkpoint_prefix_uri,
    _configure_val_audio_probe,
    _derive_probe_uri,
)
from synth_setter.data.vst import param_specs
from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io
from synth_setter.utils.callbacks import ValAudioProbe

_LAUNCH_NAMESPACE = f"train-20260720T000000000Z-{'0' * 32}"


@pytest.fixture(autouse=True)
def _skip_r2_auth_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the R2 pre-flight so these tests need no credentials.

    :param monkeypatch: Replaces the auth ping with a no-op.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", lambda *_args, **_kwargs: None)


def _cfg(
    *,
    enabled: bool | Literal["auto"],
    with_render: bool = True,
    with_synth: bool = True,
    output_dir: str = "/runs/out",
) -> DictConfig:
    """Build the minimal train cfg slice ``_configure_val_audio_probe`` reads.

    :param enabled: Value for ``training.val_audio_probe``.
    :param with_render: When ``False``, omit the ``render`` group entirely.
    :param with_synth: When ``False``, omit the root ``synth`` group (#2565).
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
    if with_synth:
        with open_dict(cfg):
            cfg.synth = {
                "name": "surge_xt",
                "param_spec_name": "surge_xt",
                "plugin_state_path": "presets/surge-base.vstpreset",
                "plugin_path": "plugins/Surge XT.vst3",
                "synth_version": "1.3.4",
            }
    if with_render:
        with open_dict(cfg):
            cfg.render = {
                "sample_rate": 44100,
                "channels": 2,
                "velocity": 100,
                "signal_duration_seconds": 4.0,
                "min_loudness": -55.0,
                "samples_per_render_batch": 1,
                "samples_per_shard": 1,
                "gui_toggle_cadence": "never",
            }
    return cfg


def test_configure_val_audio_probe_appends_nothing_when_disabled() -> None:
    """False leaves the callback list untouched."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=False), callbacks, _LAUNCH_NAMESPACE)

    assert callbacks == []


def test_configure_val_audio_probe_appends_nothing_when_setting_absent() -> None:
    """A legacy config without the probe setting leaves callbacks untouched."""
    callbacks: list[Callback] = []
    cfg = _cfg(enabled=False)
    with open_dict(cfg):
        del cfg.training.val_audio_probe

    _configure_val_audio_probe(cfg, callbacks, _LAUNCH_NAMESPACE)

    assert callbacks == []


def test_configure_val_audio_probe_appends_probe_when_enabled() -> None:
    """Enabling the flag wires exactly one ValAudioProbe under the run's output dir."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=True), callbacks, _LAUNCH_NAMESPACE)

    assert len(callbacks) == 1
    probe = callbacks[0]
    assert isinstance(probe, ValAudioProbe)
    assert probe.num_samples == 5
    assert probe.probe_root == Path("/runs/out") / "val_audio_probe"


def test_configure_val_audio_probe_forwards_validated_render_config() -> None:
    """The callback receives both synth identity and renderer settings."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=True), callbacks, _LAUNCH_NAMESPACE)

    probe = callbacks[0]
    assert isinstance(probe, ValAudioProbe)
    settings = probe._probe_fn.keywords["settings"]  # noqa: SLF001
    assert settings.param_spec_name == "surge_xt"
    assert settings.plugin_state_path == "presets/surge-base.vstpreset"
    assert settings.synth.synth_version == "1.3.4"
    assert settings.sample_rate == 44100


def test_configure_val_audio_probe_derives_torchsynth_note_params_from_datamodule() -> None:
    """TorchSynth probe rows receive encoded fixed note values from the online data config."""
    cfg = _cfg(enabled=True)
    with open_dict(cfg):
        cfg.synth = {
            "name": "torchsynth_full",
            "param_spec_name": "torchsynth_full",
            "plugin_state_path": "",
            "plugin_path": "torchsynth",
            "synth_version": "1.0.2",
        }
        cfg.datamodule = {
            "midi_pitch": 66,
            "sample_rate": 44_100,
            "signal_length": 88_200,
        }
        cfg.render.renderer_backend = "torchsynth"
    callbacks: list[Callback] = []

    _configure_val_audio_probe(cfg, callbacks, _LAUNCH_NAMESPACE)

    probe = callbacks[0]
    assert isinstance(probe, ValAudioProbe)
    assert probe.complete_rows is not None
    spec = param_specs[ParamSpecName("torchsynth_full")]
    full_row = probe.complete_rows(torch.zeros(1, spec.synth_param_length))
    _, note_params = decode_model_output(full_row[0].numpy(), spec)
    assert note_params == {"pitch": 66, "note_start_and_end": (0.0, 2.0)}


def test_configure_val_audio_probe_raises_when_render_group_missing() -> None:
    """Enabling the probe without a render group fails with a directed error."""
    with pytest.raises(ValueError, match="render"):
        _configure_val_audio_probe(_cfg(enabled=True, with_render=False), [], _LAUNCH_NAMESPACE)


def test_probe_uri_isolates_independent_same_config_launches() -> None:
    """Separate launches of one config archive probes under separate namespaces."""
    cfg = _cfg(enabled=True)
    first_namespace = f"train-20260715T000000000Z-{'0' * 31}1"
    second_namespace = f"train-20260715T000001000Z-{'0' * 31}2"

    first = _derive_probe_uri(cfg, first_namespace)
    second = _derive_probe_uri(cfg, second_namespace)

    assert first == f"r2://intermediate-data/probes/train/{first_namespace}"
    assert second == f"r2://intermediate-data/probes/train/{second_namespace}"
    assert first != second


def test_probe_uri_resume_uses_new_launch_namespace_for_recovered_run() -> None:
    """A resumed W&B run preserves its ID but starts a new probe launch namespace."""
    cfg = _cfg(enabled=True)
    recovered_run_id = "train-20260715T000000000Z"

    source = _derive_probe_uri(cfg, f"{recovered_run_id}-{'0' * 31}1")
    resumed = _derive_probe_uri(cfg, f"{recovered_run_id}-{'0' * 31}2")

    assert source != resumed
    assert f"/{recovered_run_id}-" in source
    assert f"/{recovered_run_id}-" in resumed


def test_derive_probe_uri_shares_namespace_segment_with_checkpoint_prefix() -> None:
    """One launch namespace names both probe and recovery-checkpoint prefixes."""
    cfg = _cfg(enabled=True)

    probe_uri = _derive_probe_uri(cfg, _LAUNCH_NAMESPACE)
    checkpoint_prefix = _checkpoint_prefix_uri(cfg, _LAUNCH_NAMESPACE)

    assert probe_uri.endswith(f"/{_LAUNCH_NAMESPACE}")
    assert checkpoint_prefix.endswith(f"/{_LAUNCH_NAMESPACE}")


def test_configure_val_audio_probe_namespaces_upload_uri_without_durability() -> None:
    """Probe namespacing does not depend on checkpoint durability."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=True), callbacks, _LAUNCH_NAMESPACE)

    probe = callbacks[0]
    assert isinstance(probe, ValAudioProbe)
    upload_uri = probe._probe_fn.keywords["upload_uri"]  # noqa: SLF001
    assert upload_uri == f"r2://intermediate-data/probes/train/{_LAUNCH_NAMESPACE}"


@pytest.mark.parametrize(
    "bad_samples", [0, -1, 2.5, None], ids=["zero", "negative", "float", "null"]
)
def test_configure_val_audio_probe_rejects_non_positive_int_samples(bad_samples: object) -> None:
    """A non-positive-integer sample count fails with a directed error, not a mid-run crash.

    :param bad_samples: Invalid ``training.val_audio_probe_samples`` override.
    """
    cfg = _cfg(enabled=True)
    cfg.training.val_audio_probe_samples = bad_samples

    with pytest.raises(ValueError, match="positive integer"):
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)


def test_configure_val_audio_probe_raises_when_synth_group_missing() -> None:
    """A render group without the root synth identity fails with a directed error (#2565)."""
    with pytest.raises(ValueError, match="synth=<name>"):
        _configure_val_audio_probe(_cfg(enabled=True, with_synth=False), [], _LAUNCH_NAMESPACE)


def test_configure_val_audio_probe_rejects_synth_missing_param_spec_name() -> None:
    """A malformed synth identity fails before probe construction."""
    cfg = _cfg(enabled=True)
    with open_dict(cfg):
        del cfg.synth.param_spec_name

    with pytest.raises(ValidationError, match="param_spec_name"):
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)


def test_configure_val_audio_probe_rejects_disabled_validation() -> None:
    """Probe on + `trainer.limit_val_batches=0` fails loudly instead of silently never firing.

    A validation-hooked probe wired into a validation-disabled run would stage nothing forever.
    """
    cfg = _cfg(enabled=True)
    with open_dict(cfg):
        cfg.trainer = {"limit_val_batches": 0}

    with pytest.raises(ValueError, match="limit_val_batches"):
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)


def test_configure_val_audio_probe_auto_wires_probe_with_render_group() -> None:
    """``auto`` behaves like ``true`` when a render group is composed."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled="auto"), callbacks, _LAUNCH_NAMESPACE)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ValAudioProbe)


def test_configure_val_audio_probe_auto_skips_without_render_group() -> None:
    """``auto`` with no render group skips the probe instead of failing the launch."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(
        _cfg(enabled="auto", with_render=False), callbacks, _LAUNCH_NAMESPACE
    )

    assert callbacks == []


def test_configure_val_audio_probe_auto_skip_warns_operator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``auto`` reports an unwired probe at warning level.

    :param caplog: Captures the operator-visible warning.
    """
    with caplog.at_level(logging.WARNING):
        _configure_val_audio_probe(_cfg(enabled="auto", with_render=False), [], _LAUNCH_NAMESPACE)

    assert any("no render group composed" in message for message in caplog.messages)


def test_configure_val_audio_probe_auto_skips_when_validation_disabled() -> None:
    """``auto`` with ``limit_val_batches=0`` skips the probe instead of failing."""
    callbacks: list[Callback] = []
    cfg = _cfg(enabled="auto")
    with open_dict(cfg):
        cfg.trainer = {"limit_val_batches": 0}

    _configure_val_audio_probe(cfg, callbacks, _LAUNCH_NAMESPACE)

    assert callbacks == []


def test_configure_val_audio_probe_auto_rejects_missing_synth() -> None:
    """``auto`` still fails fast without identity — a composed render group is intent."""
    cfg = _cfg(enabled="auto", with_synth=False)

    with pytest.raises(ValueError, match="synth"):
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)


@pytest.mark.parametrize(
    "mode",
    [None, "", 0, 1, "yes"],
    ids=["null", "empty", "zero", "one", "unknown-string"],
)
def test_configure_val_audio_probe_rejects_unknown_mode(mode: object) -> None:
    """A value outside true/false/auto fails with a directed error.

    :param mode: Unsupported probe-mode value.
    """
    cfg = _cfg(enabled=False)
    cfg.training.val_audio_probe = mode

    with pytest.raises(ValueError, match="auto"):
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)


def _no_r2() -> None:
    """Raise like ``ensure_r2_env_loaded`` on a credential-less host.

    :raises RuntimeError: Always.
    """
    raise RuntimeError("R2 credentials missing")


def test_configure_val_audio_probe_auto_skips_when_r2_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` on a host without R2 credentials skips the probe instead of failing.

    :param monkeypatch: Makes the R2 pre-flight raise like a credential-less host.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _no_r2)
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled="auto"), callbacks, _LAUNCH_NAMESPACE)

    assert callbacks == []


def test_configure_val_audio_probe_true_propagates_r2_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``true`` keeps the R2 pre-flight fatal.

    :param monkeypatch: Makes the R2 pre-flight raise like a credential-less host.
    """
    monkeypatch.setattr(r2_io, "ensure_r2_env_loaded", _no_r2)

    with pytest.raises(RuntimeError, match="R2 credentials missing"):
        _configure_val_audio_probe(_cfg(enabled=True), [], _LAUNCH_NAMESPACE)
