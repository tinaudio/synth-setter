"""Standalone #489 reproducer: render one identical patch N times and score all-pairs MSS.

Measures the bug at its source, bypassing the oracle/sweep harness that masks it.
For each reuse depth N it renders the *same* patch N times against a single cached
``VST3Plugin`` (the ``plugin_reload_cadence="once"`` path) and reports the
**all-pairs worst-case** multi-scale spectral loss across those N renders — the
exact signal that defined #489 in the PR #706 reproducer. Identical params should
render identically, so any worst pair above the threshold is render-to-render junk.

A control arm re-renders the same patch reloading the plugin per call (the #713
workaround, now the default), so a run that prints reuse≫threshold while
reload<threshold confirms the bug is still live and merely worked around.

The harness sweeps (``synth_setter.tools.cadence_sweep_489``) could not surface
this: they score target-vs-oracle-prediction *means*, never two raw renders of one
patch against each other.

Run it::

    python -m synth_setter.tools.repro_489_reuse_depth                 # depths 12,40 + control
    python -m synth_setter.tools.repro_489_reuse_depth --depths 12 80 --wandb
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from loguru import logger
from pyloudnorm import Meter

from synth_setter.data.vst import param_specs, preset_paths
from synth_setter.data.vst.core import load_plugin, load_preset, render_params
from synth_setter.data.vst.param_spec import NoteParams
from synth_setter.data.vst.param_spec_registry import default_plugin_path
from synth_setter.evaluation.compute_audio_metrics import compute_mss

# Render settings mirror configs/render/surge_xt.yaml so the reproduction matches
# the production render path the #489 datasets are generated with.
_SAMPLE_RATE = 44100
_CHANNELS = 2
_VELOCITY = 100
_DURATION_SECONDS = 4.0
_MIN_LOUDNESS_DB = -55.0

# PR #706 threshold: the per-pair MSS of identical params sits well under this; the
# junk render pushed the worst pair to ~21, ~6x the clean per-pair value.
_MSS_THRESHOLD = 10.0
_DEFAULT_DEPTHS = (12, 40)
# Guards the loud-patch search from spinning forever on a silent param spec.
_MAX_PATCH_DRAWS = 200

# Verdict labels for ``classify``.
BUG_PRESENT = "BUG_PRESENT"
NO_REPRO = "NO_REPRO"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ReuseDepthResult:
    """All-pairs scoring of N renders of one identical patch.

    .. attribute :: depth

       Number of renders compared (the reuse depth).

    .. attribute :: mss_max

       Worst-case multi-scale spectral loss over all unordered pairs.

    .. attribute :: worst_pair

       ``(i, j)`` render indices that produced ``mss_max``.

    .. attribute :: pair_count

       Number of unordered pairs scored, ``depth*(depth-1)/2``.

    .. attribute :: peaks

       Per-render peak amplitude; a near-zero entry is a silent junk render.
    """

    depth: int
    mss_max: float
    worst_pair: tuple[int, int]
    pair_count: int
    peaks: list[float]


def all_pairs_worst_mss(
    renders: list[np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float] = compute_mss,
) -> ReuseDepthResult:
    """Score every unordered pair of renders and return the worst-case divergence.

    :param renders: Audio clips, each shape ``(C, T)``, all rendered from one patch.
    :param metric: Pairwise distance; defaults to :func:`compute_mss`.
    :returns: The worst pair, its score, the pair count, and per-render peaks.
    :raises ValueError: Fewer than two renders — no pair to compare.
    """
    n = len(renders)
    if n < 2:
        raise ValueError(f"all-pairs scoring needs at least 2 renders, got {n}")
    worst_score, worst_i, worst_j = -1.0, -1, -1
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


def classify(reused_max: float, reloaded_max: float, threshold: float) -> str:
    """Label a depth's outcome from its reuse and reload all-pairs worst scores.

    :param reused_max: Worst all-pairs MSS on the cached-plugin (reuse) arm.
    :param reloaded_max: Worst all-pairs MSS on the per-render-reload control arm.
    :param threshold: Score above which identical-patch renders count as junk.
    :returns: ``BUG_PRESENT`` (reuse diverges, reload holds), ``INCONCLUSIVE``
        (both diverge — patch/phase, not the reuse bug), or ``NO_REPRO``.
    """
    if reused_max <= threshold:
        return NO_REPRO
    return INCONCLUSIVE if reloaded_max > threshold else BUG_PRESENT


def _integrated_loudness(audio: np.ndarray) -> float:
    """Return ITU integrated loudness (LUFS) of a ``(C, T)`` clip.

    :param audio: Rendered clip, shape ``(C, T)``.
    :returns: Integrated loudness in LUFS.
    """
    return float(Meter(_SAMPLE_RATE).integrated_loudness(audio.T))


def _sample_loud_patch(
    spec_name: str, plugin_path: str, preset_path: str
) -> tuple[dict[str, float], NoteParams]:
    """Draw one patch that renders above ``_MIN_LOUDNESS_DB``, reused across all depths.

    Renders each candidate once via the reload path (a fresh plugin) so the gate
    decision is never itself contaminated by reuse-state. Reuses one patch for the
    whole run so depths and arms are directly comparable.

    :param spec_name: Param-spec registry key (e.g. ``"surge_xt"``).
    :param plugin_path: VST3 bundle path.
    :param preset_path: Preset applied before each render.
    :returns: The ``(synth_params, note_params)`` tuple of the first loud draw.
    :raises RuntimeError: No draw cleared the gate within ``_MAX_PATCH_DRAWS``.
    """
    spec = param_specs[spec_name]
    for draw in range(_MAX_PATCH_DRAWS):
        synth_params, note_params = spec.sample()
        audio = render_params(
            plugin_path,
            synth_params,
            note_params["pitch"],
            _VELOCITY,
            note_params["note_start_and_end"],
            _DURATION_SECONDS,
            _SAMPLE_RATE,
            _CHANNELS,
            preset_path=preset_path,
        )
        loudness = _integrated_loudness(audio)
        if loudness >= _MIN_LOUDNESS_DB:
            logger.info(f"loud patch found on draw {draw} ({loudness:.1f} LUFS)")
            return synth_params, note_params
        logger.debug(f"draw {draw}: {loudness:.1f} LUFS below gate, redrawing")
    raise RuntimeError(
        f"no patch cleared {_MIN_LOUDNESS_DB} dB in {_MAX_PATCH_DRAWS} draws for {spec_name!r}"
    )


def render_identical_patch(
    plugin_path: str,
    preset_path: str,
    synth_params: dict[str, float],
    note_params: NoteParams,
    depth: int,
    *,
    reuse_plugin: bool,
) -> list[np.ndarray]:
    """Render one patch ``depth`` times, either reusing one instance or reloading per call.

    :param plugin_path: VST3 bundle path.
    :param preset_path: Preset applied to the plugin.
    :param synth_params: Synth patch held identical across every render.
    :param note_params: Note params (``pitch``, ``note_start_and_end``) held identical.
    :param depth: Number of renders (the reuse depth).
    :param reuse_plugin: When True, load + preset once and reuse the cached instance
        (the ``plugin_reload_cadence="once"`` bug path); when False, reload per render
        (the #713 workaround / current default).
    :returns: ``depth`` rendered clips, each shape ``(C, T)``.
    """
    cached = None
    if reuse_plugin:
        cached = load_plugin(plugin_path)
        load_preset(cached, preset_path)
    return [
        render_params(
            plugin_path,
            synth_params,
            note_params["pitch"],
            _VELOCITY,
            note_params["note_start_and_end"],
            _DURATION_SECONDS,
            _SAMPLE_RATE,
            _CHANNELS,
            preset_path=preset_path,
            plugin=cached,
        )
        for _ in range(depth)
    ]


def run(
    depths: tuple[int, ...],
    *,
    spec_name: str,
    control: bool,
    wandb_enabled: bool,
) -> dict[int, str]:
    """Reproduce #489 at each depth and report all-pairs worst MSS for reuse vs reload.

    :param depths: Reuse depths to probe, in order.
    :param spec_name: Param-spec registry key selecting synth + preset.
    :param control: Also render the per-render-reload control arm at each depth.
    :param wandb_enabled: Log per-depth all-pairs metrics to W&B.
    :returns: Map of depth to its :data:`classify` verdict.
    """
    plugin_path = default_plugin_path()
    preset_path = preset_paths[spec_name]
    synth_params, note_params = _sample_loud_patch(spec_name, plugin_path, preset_path)

    wandb_run = _init_wandb(spec_name, depths) if wandb_enabled else None
    verdicts: dict[int, str] = {}
    for depth in depths:
        reused = all_pairs_worst_mss(
            render_identical_patch(
                plugin_path, preset_path, synth_params, note_params, depth, reuse_plugin=True
            )
        )
        reloaded_max = float("nan")
        if control:
            reloaded = all_pairs_worst_mss(
                render_identical_patch(
                    plugin_path, preset_path, synth_params, note_params, depth, reuse_plugin=False
                )
            )
            reloaded_max = reloaded.mss_max
        verdict = (
            classify(reused.mss_max, reloaded_max, _MSS_THRESHOLD)
            if control
            else (BUG_PRESENT if reused.mss_max > _MSS_THRESHOLD else NO_REPRO)
        )
        verdicts[depth] = verdict
        _report_depth(depth, reused, reloaded_max, verdict)
        if wandb_run is not None:
            _log_depth_to_wandb(wandb_run, depth, reused, reloaded_max)
    if wandb_run is not None:
        wandb_run.finish()
    return verdicts


def _report_depth(depth: int, reused: ReuseDepthResult, reloaded_max: float, verdict: str) -> None:
    """Print one depth's reuse/reload worst-pair scores, peaks, and verdict.

    :param depth: Reuse depth being reported.
    :param reused: All-pairs result of the cached-plugin arm.
    :param reloaded_max: Worst all-pairs MSS of the reload control (``nan`` when skipped).
    :param verdict: This depth's :func:`classify` label.
    """
    logger.info(
        f"depth={depth:>3} | reuse worst mss={reused.mss_max:7.3f} @ {reused.worst_pair} "
        f"| reload worst mss={reloaded_max:7.3f} | {verdict}"
    )
    logger.info(f"depth={depth:>3} | reuse peaks={[round(p, 4) for p in reused.peaks]}")


def _init_wandb(spec_name: str, depths: tuple[int, ...]) -> Any:
    """Start a W&B run for the reproduction; import is local so non-W&B runs stay light.

    :param spec_name: Param-spec registry key, logged to the run config.
    :param depths: Reuse depths probed, logged to the run config.
    :returns: The initialised ``wandb`` run handle.
    """
    import wandb

    return wandb.init(
        project="synth-setter",
        job_type="repro-489-reuse-depth",
        config={
            "spec_name": spec_name,
            "depths": list(depths),
            "mss_threshold": _MSS_THRESHOLD,
            "min_loudness_db": _MIN_LOUDNESS_DB,
        },
    )


def _log_depth_to_wandb(
    wandb_run: Any, depth: int, reused: ReuseDepthResult, reloaded_max: float
) -> None:
    """Log a depth's all-pairs worst MSS (reuse and reload) and min peak to W&B.

    :param wandb_run: Active W&B run handle from :func:`_init_wandb`.
    :param depth: Reuse depth being logged (the x-axis).
    :param reused: All-pairs result of the cached-plugin arm.
    :param reloaded_max: Worst all-pairs MSS of the reload control (``nan`` when skipped).
    """
    wandb_run.log(
        {
            "depth": depth,
            "reuse-once/all-pairs-mss-max": reused.mss_max,
            "reload-render/all-pairs-mss-max": reloaded_max,
            "reuse-once/min-peak-amplitude": min(reused.peaks),
        }
    )


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

    verdicts = run(
        tuple(args.depths),
        spec_name=args.spec,
        control=not args.no_control,
        wandb_enabled=args.wandb,
    )
    if BUG_PRESENT in verdicts.values():
        depths = [d for d, v in verdicts.items() if v == BUG_PRESENT]
        logger.warning(f"#489 reproduced at depths {depths}: reuse junk, reload clean")
        sys.exit(1)


if __name__ == "__main__":
    main()
