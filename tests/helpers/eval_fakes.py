"""Shared fakes for eval-entrypoint postprocessing tests."""

from collections.abc import Callable
from pathlib import Path

PREDICT_VST_AUDIO_FRAGMENT = "predict_vst_audio"
COMPUTE_AUDIO_METRICS_FRAGMENT = "compute_audio_metrics"

FAKE_AGGREGATED_METRICS_CSV = (
    ",mean,std\nmss,0.5,0.1\nwmfcc,0.3,0.05\nsot,0.2,0.02\nrms,0.9,0.01\n"
)


def fake_metrics_csv(num_samples: int = 2) -> str:
    """Build a per-sample metrics CSV with ``num_samples`` rows.

    :param num_samples: Number of sample rows to emit.
    :returns: CSV body matching ``compute_audio_metrics``'s per-sample output shape.
    """
    rows = [f"{idx},0.1,0.2,0.3,0.4" for idx in range(num_samples)]
    return ",mss,wmfcc,sot,rms\n" + "\n".join(rows) + "\n"


FAKE_METRICS_CSV = fake_metrics_csv()


def fake_postprocessing_subprocess(
    *,
    audio_metrics_csv: str = FAKE_AGGREGATED_METRICS_CSV,
    per_sample_metrics_csv: str | None = None,
    shuffled_metrics_csv: str | None = None,
    shuffle_permutation_csv: str | None = None,
    render_sample_count: int = 0,
) -> Callable[[list[str]], None]:
    """Build a ``subprocess.run`` fake that materializes eval postprocessing outputs.

    :param audio_metrics_csv: Body written to ``aggregated_metrics.csv``.
    :param per_sample_metrics_csv: Optional body written to ``metrics.csv``.
    :param shuffled_metrics_csv: Optional body written to ``aggregated_metrics_shuffled.csv``.
    :param shuffle_permutation_csv: Optional body written to ``shuffle_permutation.csv``.
    :param render_sample_count: Number of fake ``sample_N`` render directories to create.
    :returns: Callable compatible with ``subprocess.run``.
    """

    def _fake_run(args: list[str], **_kwargs: object) -> None:
        is_render = any(PREDICT_VST_AUDIO_FRAGMENT in arg for arg in args)
        is_metrics = any(COMPUTE_AUDIO_METRICS_FRAGMENT in arg for arg in args)
        if not (is_render or is_metrics):
            return

        output_dir = Path(args[args.index("-m") + 3])
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_render:
            _write_render_outputs(output_dir, render_sample_count)
        if is_metrics:
            _write_metrics_outputs(
                output_dir,
                audio_metrics_csv=audio_metrics_csv,
                per_sample_metrics_csv=per_sample_metrics_csv,
                shuffled_metrics_csv=shuffled_metrics_csv,
                shuffle_permutation_csv=shuffle_permutation_csv,
            )

    return _fake_run


def _write_render_outputs(output_dir: Path, sample_count: int) -> None:
    """Write the fake render tree consumed by metrics postprocessing tests.

    :param output_dir: Render output directory.
    :param sample_count: Number of sample directories to create.
    """
    for sample_idx in range(sample_count):
        sample_dir = output_dir / f"sample_{sample_idx}"
        sample_dir.mkdir()
        for name in ("target.wav", "pred.wav", "spec.png", "params.csv"):
            (sample_dir / name).write_text("fake")


def _write_metrics_outputs(
    output_dir: Path,
    *,
    audio_metrics_csv: str,
    per_sample_metrics_csv: str | None,
    shuffled_metrics_csv: str | None,
    shuffle_permutation_csv: str | None,
) -> None:
    """Write the fake metrics files requested by a test.

    :param output_dir: Metrics output directory.
    :param audio_metrics_csv: Body written to ``aggregated_metrics.csv``.
    :param per_sample_metrics_csv: Optional body written to ``metrics.csv``.
    :param shuffled_metrics_csv: Optional body written to ``aggregated_metrics_shuffled.csv``.
    :param shuffle_permutation_csv: Optional body written to ``shuffle_permutation.csv``.
    """
    (output_dir / "aggregated_metrics.csv").write_text(audio_metrics_csv)
    if per_sample_metrics_csv is not None:
        (output_dir / "metrics.csv").write_text(per_sample_metrics_csv)
    if shuffled_metrics_csv is not None:
        (output_dir / "aggregated_metrics_shuffled.csv").write_text(shuffled_metrics_csv)
    if shuffle_permutation_csv is not None:
        (output_dir / "shuffle_permutation.csv").write_text(shuffle_permutation_csv)
