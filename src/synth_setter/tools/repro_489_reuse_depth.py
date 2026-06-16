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
from typing import TYPE_CHECKING

import numpy as np
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

# Render settings mirror configs/render/surge_xt.yaml so the reproduction matches
# the production render path the #489 datasets are generated with.
_SAMPLE_RATE = 44100
_CHANNELS = 2
_VELOCITY = 100
_DURATION_SECONDS = 4.0
_MIN_LOUDNESS_DB = -55.0

# PR #706 threshold: identical-param renders sit well under this; the junk render
# pushed the worst pair to ~21, ~6x the clean per-pair value.
_MSS_THRESHOLD = 10.0
_DEFAULT_DEPTHS = (12, 40)
# Guards the loud-patch search from spinning forever on a silent param spec.
_MAX_PATCH_DRAWS = 200


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
class PatchSpec:
    """One synth patch plus the plugin and preset that render it.

    .. attribute :: plugin_path

       VST3 bundle path.

    .. attribute :: preset_path

       Preset applied before each render.

    .. attribute :: synth_params

       Synth patch held identical across every render.

    .. attribute :: note_params

       ``pitch`` and ``note_start_and_end`` held identical across every render.
    """

    plugin_path: str
    preset_path: str
    synth_params: dict[str, float]
    note_params: NoteParams


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

    :param renders: Audio clips, each shape ``(C, T)``, all rendered from one patch.
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


def _integrated_loudness(audio: np.ndarray) -> float:
    """Return ITU integrated loudness (LUFS) of a ``(C, T)`` clip.

    :param audio: Rendered clip, shape ``(C, T)``.
    :returns: Integrated loudness in LUFS.
    """
    return float(Meter(_SAMPLE_RATE).integrated_loudness(audio.T))


def resolve_patch(spec_name: str) -> PatchSpec:
    """Draw one patch that renders above ``_MIN_LOUDNESS_DB`` for the named spec.

    Renders each candidate once via the reload path (a fresh plugin) so the gate
    decision is never itself contaminated by reuse-state.

    :param spec_name: Param-spec registry key (e.g. ``"surge_xt"``).
    :returns: The first loud draw, bound to its plugin and preset.
    :raises RuntimeError: No draw cleared the gate within ``_MAX_PATCH_DRAWS``.
    """
    plugin_path = default_plugin_path()
    preset_path = preset_paths[spec_name]
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
            logger.info("loud patch found on draw %d (%.1f LUFS)", draw, loudness)
            return PatchSpec(plugin_path, preset_path, synth_params, note_params)
        logger.debug("draw %d: %.1f LUFS below gate, redrawing", draw, loudness)
    raise RuntimeError(
        f"no patch cleared {_MIN_LOUDNESS_DB} dB in {_MAX_PATCH_DRAWS} draws for {spec_name!r}"
    )


def _render_repeated(
    patch: PatchSpec, depth: int, cached_plugin: VST3Plugin | None
) -> list[np.ndarray]:
    """Render ``patch`` ``depth`` times, reusing ``cached_plugin`` when supplied.

    :param patch: The identical patch and its plugin/preset.
    :param depth: Number of renders (the reuse depth).
    :param cached_plugin: Reused instance for the #489 ``once`` arm, or ``None`` to
        reload a fresh plugin per render (the #713 reload arm / current default).
    :returns: ``depth`` rendered clips, each shape ``(C, T)``.
    """
    return [
        render_params(
            patch.plugin_path,
            patch.synth_params,
            patch.note_params["pitch"],
            _VELOCITY,
            patch.note_params["note_start_and_end"],
            _DURATION_SECONDS,
            _SAMPLE_RATE,
            _CHANNELS,
            preset_path=patch.preset_path,
            plugin=cached_plugin,
        )
        for _ in range(depth)
    ]


def _render_reusing_one_plugin(patch: PatchSpec, depth: int) -> list[np.ndarray]:
    """Render ``patch`` ``depth`` times against one cached instance (the #489 ``once`` arm).

    :param patch: The identical patch and its plugin/preset.
    :param depth: Number of renders sharing the cached instance.
    :returns: ``depth`` rendered clips, each shape ``(C, T)``.
    """
    plugin = load_plugin(patch.plugin_path)
    load_preset(plugin, patch.preset_path)
    return _render_repeated(patch, depth, cached_plugin=plugin)


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
    for depth in depths:
        reused = all_pairs_worst_mss(_render_reusing_one_plugin(patch, depth))
        # Control arm reloads a fresh plugin per render (cached_plugin=None) — the #713
        # workaround / current default; skipped under --no-control.
        reloaded_max = (
            all_pairs_worst_mss(_render_repeated(patch, depth, cached_plugin=None)).mss_max
            if control
            else None
        )
        verdict = classify(reused.mss_max, reloaded_max, _MSS_THRESHOLD)
        verdicts[depth] = verdict
        _report_depth(depth, reused, reloaded_max, verdict)
        if wandb_run is not None:
            _log_depth_to_wandb(wandb_run, depth, reused, reloaded_max)
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

    :param spec_name: Param-spec registry key.
    :param depths: Reuse depths probed.
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
            "min_loudness_db": _MIN_LOUDNESS_DB,
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
