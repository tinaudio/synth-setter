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
from lightning import Callback
from omegaconf import DictConfig, OmegaConf, open_dict

from synth_setter.cli.train import (
    _checkpoint_prefix_uri,
    _configure_val_audio_probe,
    _derive_probe_uri,
)
from synth_setter.pipeline import r2_io
from synth_setter.utils.callbacks import ValAudioProbe

# Shaped like _make_recovery_namespace output: "{run_id}-{uuid4().hex}".
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
    output_dir: str = "/runs/out",
    datamodule: dict[str, str | None] | None = None,
) -> DictConfig:
    """Build the minimal train cfg slice ``_configure_val_audio_probe`` reads.

    :param enabled: Value for ``training.val_audio_probe``.
    :param with_render: When ``False``, omit the ``render`` group entirely.
    :param output_dir: Value for ``paths.output_dir``.
    :param datamodule: Optional ``datamodule`` group; ``None`` omits it entirely.
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
    if datamodule is not None:
        with open_dict(cfg):
            cfg.datamodule = datamodule
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


def test_configure_val_audio_probe_raises_when_render_group_missing() -> None:
    """Enabling the probe without a render group fails with a directed error."""
    with pytest.raises(ValueError, match="render"):
        _configure_val_audio_probe(_cfg(enabled=True, with_render=False), [], _LAUNCH_NAMESPACE)


def test_derive_probe_uri_places_namespace_between_config_id_and_steps() -> None:
    """Snapshots archive under probes/{config_id}/{launch namespace} (#2230)."""
    uri = _derive_probe_uri(_cfg(enabled=True), _LAUNCH_NAMESPACE)

    assert uri == f"r2://intermediate-data/probes/train/{_LAUNCH_NAMESPACE}"


def test_derive_probe_uri_shares_namespace_segment_with_checkpoint_prefix() -> None:
    """One launch namespace names both the probe prefix and the recovery-checkpoint prefix."""
    cfg = _cfg(enabled=True)

    probe_uri = _derive_probe_uri(cfg, _LAUNCH_NAMESPACE)
    checkpoint_prefix = _checkpoint_prefix_uri(cfg, _LAUNCH_NAMESPACE)

    assert probe_uri.endswith(f"/{_LAUNCH_NAMESPACE}")
    assert checkpoint_prefix.endswith(f"/{_LAUNCH_NAMESPACE}")


def test_configure_val_audio_probe_namespaces_upload_uri_without_durability() -> None:
    """The wired upload URI carries the launch namespace even with durability off.

    The fixture cfg never sets ``training.upload_checkpoints_during_training``,
    so this pins that probe namespacing does not depend on the durability flag.
    """
    callbacks: list[Callback] = []

    _configure_val_audio_probe(_cfg(enabled=True), callbacks, _LAUNCH_NAMESPACE)

    probe = callbacks[0]
    assert isinstance(probe, ValAudioProbe)
    upload_uri = probe._probe_fn.keywords["upload_uri"]  # noqa: SLF001 — pins the wired partial
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


def test_configure_val_audio_probe_rejects_render_spec_mismatching_datamodule() -> None:
    """A render spec that cannot decode the model's output layout fails at configure time.

    A ``surge_simple`` model probed with ``render=surge_xt`` decodes 92-dim
    predictions against the 164-param spec: every probe cycle dies in the
    subprocess and the run silently produces no audio metrics (#1990).
    """
    cfg = _cfg(enabled=True, datamodule={"param_spec_name": "surge_simple"})

    with pytest.raises(ValueError) as excinfo:
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)

    assert "render.param_spec_name is 'surge_xt'" in str(excinfo.value)
    assert "datamodule.param_spec_name='surge_simple'" in str(excinfo.value)


def test_configure_val_audio_probe_accepts_render_spec_matching_datamodule() -> None:
    """A render spec matching the datamodule's spec wires the probe normally."""
    callbacks: list[Callback] = []

    _configure_val_audio_probe(
        _cfg(enabled=True, datamodule={"param_spec_name": "surge_xt"}),
        callbacks,
        _LAUNCH_NAMESPACE,
    )

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ValAudioProbe)


def test_configure_val_audio_probe_rejects_render_group_missing_spec_key() -> None:
    """A render group without ``param_spec_name`` fails and the message says it is unset."""
    cfg = _cfg(enabled=True, datamodule={"param_spec_name": "surge_simple"})
    with open_dict(cfg):
        del cfg.render.param_spec_name

    with pytest.raises(ValueError) as excinfo:
        _configure_val_audio_probe(cfg, [], _LAUNCH_NAMESPACE)

    assert "render.param_spec_name is unset" in str(excinfo.value)


@pytest.mark.parametrize(
    "datamodule",
    [{"batch_size": "8"}, {"param_spec_name": None}],
    ids=["key-absent", "key-null"],
)
def test_configure_val_audio_probe_skips_spec_check_when_datamodule_has_no_spec(
    datamodule: dict[str, str | None],
) -> None:
    """A datamodule without a ``param_spec_name`` value (non-VST) leaves the guard inert.

    :param datamodule: Datamodule group variant carrying no usable spec name.
    """
    callbacks: list[Callback] = []

    _configure_val_audio_probe(
        _cfg(enabled=True, datamodule=datamodule), callbacks, _LAUNCH_NAMESPACE
    )

    assert len(callbacks) == 1


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


def test_configure_val_audio_probe_auto_rejects_spec_mismatch() -> None:
    """``auto`` still fails fast on a decode mismatch — a composed render group is intent."""
    cfg = _cfg(enabled="auto", datamodule={"param_spec_name": "surge_simple"})

    with pytest.raises(ValueError, match="param_spec_name"):
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
