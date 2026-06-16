"""Standalone #489 reproducer: render one identical patch N times and score all-pairs MSS.

For each reuse depth N it renders the *same* patch N times against a single cached
``VST3Plugin`` (the ``plugin_reload_cadence="once"`` path) and reports the
**all-pairs worst-case** multi-scale spectral loss across those N renders — the
signal that defined #489 in the PR #706 reproducer. Identical params should render
identically, so any worst pair above the threshold is render-to-render junk.

A control arm re-renders the same patch reloading the plugin per call (the #713
workaround, now the default), so reuse>>threshold while reload<threshold pins the
bug as live and merely worked around. The oracle/sweep harness scores
target-vs-prediction means, so only a direct render-vs-render comparison like this
can surface it.

Run it::

    python -m synth_setter.tools.repro_489_reuse_depth                 # depths 12,40 + control
    python -m synth_setter.tools.repro_489_reuse_depth --depths 12 80 --wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omegaconf import OmegaConf
from pyloudnorm import Meter

from synth_setter.data.vst import param_specs, preset_paths
from synth_setter.data.vst.core import load_plugin, load_preset, render_params
from synth_setter.data.vst.param_spec import NoteParams
from synth_setter.data.vst.param_spec_registry import default_plugin_path
from synth_setter.evaluation.compute_audio_metrics import compute_mss

if TYPE_CHECKING:
    from pedalboard import VST3Plugin
    from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)

# PR #706 threshold: identical-param renders sit well under this; the junk render
# pushed the worst pair to ~21, ~6x the clean per-pair value.
_MSS_THRESHOLD = 10.0
_DEFAULT_DEPTHS = (12, 40)
# Guards the loud-patch search from spinning forever on a silent param spec.
_MAX_PATCH_DRAWS = 200
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
# configs/render/<spec>.yaml relative to this tools/ module.
_RENDER_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "render"


class Verdict(StrEnum):
    """Outcome of one reuse depth's reuse-vs-reload comparison.

    .. attribute :: BUG_PRESENT

       Reuse arm above threshold while the reload control held — #489 live.

    .. attribute :: NO_REPRO

       Reuse arm within threshold — no junk at this depth.

    .. attribute :: INCONCLUSIVE

       Both arms above threshold — divergence is patch/phase, not reuse.
    """

    BUG_PRESENT = "BUG_PRESENT"
    NO_REPRO = "NO_REPRO"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class RenderSettings:
    """Render knobs sourced from ``configs/render/<spec>.yaml`` (single source of truth).

    .. attribute :: sample_rate

       Audio sample rate in Hz.

    .. attribute :: channels

       Audio channel count.

    .. attribute :: velocity

       MIDI velocity for every render.

    .. attribute :: duration_seconds

       Render length in seconds.

    .. attribute :: min_loudness_db

       Loudness floor (LUFS) the seed patch must clear.
    """

    sample_rate: int
    channels: int
    velocity: int
    duration_seconds: float
    min_loudness_db: float


@dataclass(frozen=True)
class PatchSpec:
    """One synth patch plus everything needed to render it identically.

    .. attribute :: plugin_path

       VST3 bundle path.

    .. attribute :: preset_path

       Preset applied before each render.

    .. attribute :: synth_params

       Synth patch held identical across every render.

    .. attribute :: note_params

       ``pitch`` and ``note_start_and_end`` held identical across every render.

    .. attribute :: settings

       Render knobs (sample rate, channels, velocity, duration, loudness floor).
    """

    plugin_path: str
    preset_path: str
    synth_params: dict[str, float]
    note_params: NoteParams
    settings: RenderSettings


@dataclass(frozen=True)
class ReuseDepthResult:
    """All-pairs scoring of N renders of one identical patch.

    .. attribute :: depth

       Render count; equals ``len(renders)``.

    .. attribute :: mss_max

       Worst-case multi-scale spectral loss (dB) over all pairs.

    .. attribute :: worst_pair

       ``(i, j)`` indices of the render pair scoring ``mss_max``.

    .. attribute :: pair_count

       ``depth*(depth-1)/2``.

    .. attribute :: peaks

       Per-render peak amplitude; near-zero marks a silent junk render.
    """

    depth: int
    mss_max: float
    worst_pair: tuple[int, int]
    pair_count: int
    peaks: list[float]


def all_pairs_worst_mss(
    renders: Sequence[np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float] = compute_mss,
) -> ReuseDepthResult:
    """Score every unordered pair of renders and return the worst-case divergence.

    :param renders: Audio clips, each shape ``(C, T)`` float32, all from one patch.
    :param metric: Pairwise distance; defaults to :func:`compute_mss`.
    :returns: A :class:`ReuseDepthResult` for the ``len(renders)`` renders.
    :raises ValueError: Fewer than two renders — no pair to compare.
    """
    n = len(renders)
    if n < 2:
        raise ValueError(f"all-pairs scoring needs at least 2 renders, got {n}")
    worst_score, worst_i, worst_j = float("-inf"), -1, -1
    for i, j in combinations(range(n), 2):
        score = metric(renders[i], renders[j])
        if score > worst_score:
            worst_score, worst_i, worst_j = score, i, j
    peaks = [float(np.abs(render).max()) for render in renders]
    return ReuseDepthResult(
        depth=n,
        mss_max=float(worst_score),
        worst_pair=(worst_i, worst_j),
        pair_count=n * (n - 1) // 2,
        peaks=peaks,
    )


def classify(reused_max: float, reloaded_max: float | None, threshold: float) -> Verdict:
    """Label a depth's outcome from its reuse and reload all-pairs worst scores.

    :param reused_max: Worst all-pairs MSS on the cached-plugin (reuse) arm.
    :param reloaded_max: Worst all-pairs MSS on the reload control arm, or ``None``
        when the control arm was skipped.
    :param threshold: Score above which identical-patch renders count as junk.
    :returns: One of the :class:`Verdict` members.
    """
    if reused_max <= threshold:
        return Verdict.NO_REPRO
    if reloaded_max is not None and reloaded_max > threshold:
        return Verdict.INCONCLUSIVE
    return Verdict.BUG_PRESENT


def load_render_settings(spec_name: str) -> RenderSettings:
    """Read render knobs from ``configs/render/<spec_name>.yaml``.

    :param spec_name: Param-spec registry key, also the render config stem.
    :returns: The render settings the production pipeline uses for this spec.
    :raises FileNotFoundError: No render config exists for ``spec_name``.
    """
    config_path = _RENDER_CONFIG_DIR / f"{spec_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"no render config for spec {spec_name!r} at {config_path}")
    cfg = OmegaConf.load(config_path)
    return RenderSettings(
        sample_rate=int(cfg.sample_rate),
        channels=int(cfg.channels),
        velocity=int(cfg.velocity),
        duration_seconds=float(cfg.signal_duration_seconds),
        min_loudness_db=float(cfg.min_loudness),
    )


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Return ITU integrated loudness (LUFS) of a ``(C, T)`` clip.

    :param audio: Rendered clip, shape ``(C, T)`` float32.
    :param sample_rate: Sample rate of ``audio`` in Hz.
    :returns: Integrated loudness in LUFS.
    """
    return float(Meter(sample_rate).integrated_loudness(audio.T))


def _render_once(patch: PatchSpec, cached_plugin: VST3Plugin | None) -> np.ndarray:
    """Render ``patch`` once, on ``cached_plugin`` if given else a freshly loaded plugin.

    :param patch: The patch and its plugin/preset/settings.
    :param cached_plugin: Injected instance to reuse, or ``None`` to load fresh.
    :returns: One rendered clip, shape ``(C, T)``.
    """
    return render_params(
        patch.plugin_path,
        patch.synth_params,
        patch.note_params["pitch"],
        patch.settings.velocity,
        patch.note_params["note_start_and_end"],
        patch.settings.duration_seconds,
        patch.settings.sample_rate,
        patch.settings.channels,
        preset_path=patch.preset_path,
        plugin=cached_plugin,
    )


def render_reusing_one_plugin(patch: PatchSpec, depth: int) -> list[np.ndarray]:
    """Render ``patch`` ``depth`` times against one cached instance (the #489 ``once`` arm).

    :param patch: The patch and its plugin/preset/settings.
    :param depth: Number of renders sharing the cached instance.
    :returns: ``depth`` rendered clips, each shape ``(C, T)``.
    """
    plugin = load_plugin(patch.plugin_path)
    load_preset(plugin, patch.preset_path)
    return [_render_once(patch, plugin) for _ in range(depth)]


def render_reloading_each_render(patch: PatchSpec, depth: int) -> list[np.ndarray]:
    """Render ``patch`` ``depth`` times, reloading a fresh plugin per render (#713 default arm).

    :param patch: The patch and its plugin/preset/settings.
    :param depth: Number of independent reload-and-render cycles.
    :returns: ``depth`` rendered clips, each shape ``(C, T)``.
    """
    return [_render_once(patch, None) for _ in range(depth)]


def resolve_patch(spec_name: str) -> PatchSpec:
    """Draw one patch that renders above the spec's loudness floor.

    Renders each candidate once via a fresh plugin so the gate decision is never
    itself contaminated by reuse-state.

    :param spec_name: Param-spec registry key (e.g. ``"surge_xt"``).
    :returns: The first loud draw, bound to its plugin, preset, and settings.
    :raises RuntimeError: No draw cleared the floor within ``_MAX_PATCH_DRAWS``.
    """
    plugin_path = default_plugin_path()
    preset_path = preset_paths[spec_name]
    settings = load_render_settings(spec_name)
    spec = param_specs[spec_name]
    for draw in range(_MAX_PATCH_DRAWS):
        synth_params, note_params = spec.sample()
        patch = PatchSpec(plugin_path, preset_path, synth_params, note_params, settings)
        loudness = integrated_loudness(_render_once(patch, None), settings.sample_rate)
        if loudness >= settings.min_loudness_db:
            logger.info("loud patch found on draw %d (%.1f LUFS)", draw, loudness)
            return patch
        logger.debug("draw %d: %.1f LUFS below floor, redrawing", draw, loudness)
    raise RuntimeError(
        f"no patch cleared {settings.min_loudness_db} dB in {_MAX_PATCH_DRAWS} draws "
        f"for {spec_name!r}"
    )


def run(
    depths: Sequence[int],
    *,
    spec_name: str,
    control: bool,
    wandb_enabled: bool,
) -> dict[int, Verdict]:
    """Reproduce #489 at each depth and report all-pairs worst MSS for reuse vs reload.

    :param depths: Reuse depths to probe, in order.
    :param spec_name: Param-spec registry key selecting synth + preset.
    :param control: Also render the per-render-reload control arm at each depth.
    :param wandb_enabled: Log per-depth all-pairs metrics to W&B.
    :returns: Map of depth to its :func:`classify` verdict.
    """
    patch = resolve_patch(spec_name)
    wandb_run = _init_wandb(spec_name, depths) if wandb_enabled else None
    verdicts: dict[int, Verdict] = {}
    try:
        for depth in depths:
            reused = all_pairs_worst_mss(render_reusing_one_plugin(patch, depth))
            reloaded_max = (
                all_pairs_worst_mss(render_reloading_each_render(patch, depth)).mss_max
                if control
                else None
            )
            verdict = classify(reused.mss_max, reloaded_max, _MSS_THRESHOLD)
            verdicts[depth] = verdict
            _report_depth(depth, reused, reloaded_max, verdict)
            if wandb_run is not None:
                _log_depth_to_wandb(wandb_run, depth, reused, reloaded_max)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return verdicts


def _report_depth(
    depth: int, reused: ReuseDepthResult, reloaded_max: float | None, verdict: Verdict
) -> None:
    """Log one depth's reuse/reload worst-pair scores, peaks, and verdict.

    :param depth: Reuse depth being reported.
    :param reused: All-pairs result of the cached-plugin arm.
    :param reloaded_max: Worst all-pairs MSS of the reload control, or ``None`` if skipped
        (rendered as ``N/A`` to keep columns aligned).
    :param verdict: This depth's :func:`classify` label.
    """
    reload_str = "    N/A" if reloaded_max is None else f"{reloaded_max:7.3f}"
    logger.info(
        "depth=%3d | reuse worst mss=%7.3f @ %s | reload worst mss=%s | %s",
        depth,
        reused.mss_max,
        reused.worst_pair,
        reload_str,
        verdict,
    )
    logger.info("depth=%3d | reuse peaks=%s", depth, [round(p, 4) for p in reused.peaks])


def _init_wandb(spec_name: str, depths: Sequence[int]) -> Run:
    """Start a W&B run for the reproduction; import is local so non-W&B runs stay light.

    :param spec_name: Param-spec registry key, recorded so a run ties to its synth.
    :param depths: Reuse depths probed, recorded for the depth axis.
    :returns: The initialised W&B run handle.
    """
    import wandb

    return wandb.init(
        project="synth-setter",
        job_type="repro-489-reuse-depth",
        config={
            "spec_name": spec_name,
            "depths": list(depths),
            "mss_threshold": _MSS_THRESHOLD,
        },
    )


def _log_depth_to_wandb(
    wandb_run: Run, depth: int, reused: ReuseDepthResult, reloaded_max: float | None
) -> None:
    """Log a depth's all-pairs worst MSS (reuse, and reload when run) and min peak to W&B.

    :param wandb_run: Active W&B run handle from :func:`_init_wandb`.
    :param depth: Reuse depth being logged (the x-axis).
    :param reused: All-pairs result of the cached-plugin arm.
    :param reloaded_max: Worst all-pairs MSS of the reload control, or ``None`` if skipped.
    """
    metrics: dict[str, float] = {
        "depth": depth,
        "reuse-once/all-pairs-mss-max": reused.mss_max,
        "reuse-once/min-peak-amplitude": min(reused.peaks),
    }
    if reloaded_max is not None:
        metrics["reload-render/all-pairs-mss-max"] = reloaded_max
    wandb_run.log(metrics)


def main(argv: list[str] | None = None) -> None:
    """CLI: reproduce #489 at the chosen depths and exit non-zero if the bug is present.

    :param argv: Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(_DEFAULT_DEPTHS),
        help=f"reuse depths to probe (default: {' '.join(map(str, _DEFAULT_DEPTHS))})",
    )
    parser.add_argument(
        "--spec", default="surge_xt", help="param-spec registry key (default: surge_xt)"
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="skip the per-render-reload control arm (reuse arm only)",
    )
    parser.add_argument("--wandb", action="store_true", help="log per-depth metrics to W&B")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    verdicts = run(
        args.depths,
        spec_name=args.spec,
        control=not args.no_control,
        wandb_enabled=args.wandb,
    )
    bug_depths = [depth for depth, verdict in verdicts.items() if verdict is Verdict.BUG_PRESENT]
    if bug_depths:
        logger.warning("#489 reproduced at depths %s: reuse junk, reload clean", bug_depths)
        sys.exit(1)


if __name__ == "__main__":
    main()
