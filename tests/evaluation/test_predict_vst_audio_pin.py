"""Pin the per-sample artifact layout produced by prediction-audio rendering."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pydantic_settings import CliApp  # noqa: E402

from synth_setter.evaluation import predict_vst_audio  # noqa: E402
from synth_setter.evaluation.predict_vst_audio import main as predict_vst_audio_main  # noqa: E402
from synth_setter.param_spec_name import ParamSpecName  # noqa: E402
from synth_setter.pipeline.schemas.spec import RenderConfig  # noqa: E402
from synth_setter.synth_spec import SynthName, SynthSpec  # noqa: E402

_PARAM_SPEC_NAME = ParamSpecName("surge_simple")
_CHANNELS = 2
_SAMPLE_RATE = 8000
_SIGNAL_DURATION_SECONDS = 0.1


class _FakeRenderer:
    """Deterministic renderer for artifact-layout tests."""

    def render(
        self,
        params: dict[str, float],
        midi_note: int,
        velocity: int,
        note_start_and_end: tuple[float, float],
        *,
        warmup: bool = False,
    ) -> np.ndarray:
        """Return finite channel-leading audio with the configured test shape.

        :param params: Ignored normalized synthesizer parameters.
        :param midi_note: Ignored MIDI pitch.
        :param velocity: Ignored MIDI velocity.
        :param note_start_and_end: Ignored note window.
        :param warmup: Ignored editor warm-up request.
        :returns: Deterministic channel-leading audio.
        """
        del params, midi_note, velocity, note_start_and_end, warmup
        num_samples = int(_SAMPLE_RATE * _SIGNAL_DURATION_SECONDS)
        rng = np.random.default_rng(42)
        return rng.standard_normal((_CHANNELS, num_samples)).astype(np.float32)


def _render_config() -> RenderConfig:
    """Return the complete config transported through the process CLI.

    :returns: Validated config for the artifact-layout test renderer.
    """
    return RenderConfig(
        synth=SynthSpec(
            name=SynthName("surge_simple"),
            param_spec_name=_PARAM_SPEC_NAME,
            plugin_path="plugins/Surge XT.vst3",
            plugin_state_path="presets/surge-simple.vstpreset",
            synth_version="1.3.4",
        ),
        sample_rate=_SAMPLE_RATE,
        channels=_CHANNELS,
        velocity=100,
        signal_duration_seconds=_SIGNAL_DURATION_SECONDS,
        min_loudness=-55.0,
        samples_per_render_batch=2,
        samples_per_shard=2,
        plugin_reload_cadence="render",
        gui_toggle_cadence="never",
    )


def _invoke_predict_cli(
    pred_dir: Path,
    out_dir: Path,
    operation_flags: tuple[str, ...],
) -> None:
    """Invoke the process CLI with a serialized complete render config.

    :param pred_dir: Directory of staged prediction tensors.
    :param out_dir: Destination artifact directory.
    :param operation_flags: Operation flags enabled for the invocation.
    """
    enabled = set(operation_flags)
    argv = [str(pred_dir), str(out_dir), *CliApp.serialize(_render_config())]
    for flag in ("--rerender-target", "--no-params", "--skip-spectrogram"):
        if flag in enabled:
            argv.extend([flag, "True"])
    predict_vst_audio_main(argv)


@pytest.fixture(autouse=True)
def _patch_renderer_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace native host construction at the shared factory boundary.

    :param monkeypatch: Pytest fixture used to patch the renderer factory.
    """
    monkeypatch.setattr(
        predict_vst_audio,
        "make_audio_renderer",
        lambda _config: _FakeRenderer(),
    )


def test_cli_writes_expected_wav_layout(fixture_pred_dir: Path, tmp_path: Path) -> None:
    """Every input row produces both wavs while suppression flags remain effective.

    :param fixture_pred_dir: Session-scoped prediction-tensor directory.
    :param tmp_path: Pytest fixture providing a fresh output directory.
    """
    out_dir = tmp_path / "out"

    _invoke_predict_cli(fixture_pred_dir, out_dir, ("--skip-spectrogram", "--no-params"))

    sample_dirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == ["sample_0", "sample_1"]
    for sample in sample_dirs:
        assert (sample / "pred.wav").is_file()
        assert (sample / "target.wav").is_file()
        assert not (sample / "spec.png").exists()
        params_df = pd.read_csv(sample / "params.csv", index_col=0)
        assert bool(params_df["target"].isna().all())


def test_cli_writes_params_csv(fixture_pred_dir: Path, tmp_path: Path) -> None:
    """The default parameter path writes pred and target columns per sample.

    :param fixture_pred_dir: Session-scoped prediction-tensor directory.
    :param tmp_path: Pytest fixture providing a fresh output directory.
    """
    out_dir = tmp_path / "out"

    _invoke_predict_cli(fixture_pred_dir, out_dir, ("--skip-spectrogram",))

    for sample_name in ("sample_0", "sample_1"):
        csv_path = out_dir / sample_name / "params.csv"
        assert csv_path.is_file()
        assert list(pd.read_csv(csv_path, index_col=0).columns) == ["pred", "target"]


def test_cli_writes_spectrogram_when_enabled(fixture_pred_dir: Path, tmp_path: Path) -> None:
    """The default spectrogram path writes a valid PNG per sample.

    :param fixture_pred_dir: Session-scoped prediction-tensor directory.
    :param tmp_path: Pytest fixture providing a fresh output directory.
    """
    out_dir = tmp_path / "out"

    _invoke_predict_cli(fixture_pred_dir, out_dir, ("--no-params",))

    for sample_name in ("sample_0", "sample_1"):
        png = out_dir / sample_name / "spec.png"
        assert png.is_file()
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
