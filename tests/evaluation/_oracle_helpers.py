"""Shared oracle-invariant fixtures: the audio-metric bounds and the module builder.

The ``surge/fake_oracle`` model returns ``batch["params"]`` verbatim, so its
rendered audio matches the target up to Surge XT's per-voice render jitter
(oscillator phase, noise seed). :data:`ORACLE_AUDIO_METRIC_BOUNDS` is the
loosest envelope that still fails on a real regression; the per-sample
assertions in ``tests/test_train.py`` read their thresholds from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from synth_setter.models.surge_fake_oracle_module import (
    FakeOracleNet,
    SurgeFakeOracleModule,
)


@dataclass(frozen=True)
class OracleAudioMetricBounds:
    """Per-metric pass thresholds; distances are upper bounds, ``rms`` is a lower bound.

    .. attribute :: mss_max

        Multi-scale-spectral distance upper bound (strict ``<``).

    .. attribute :: wmfcc_max

        Weighted-MFCC distance upper bound (strict ``<``).

    .. attribute :: sot_max

        Sum-of-transients distance upper bound (strict ``<``).

    .. attribute :: rms_min

        RMS-envelope cosine-similarity lower bound (strict ``>``).
    """

    mss_max: float
    wmfcc_max: float
    sot_max: float
    rms_min: float


ORACLE_AUDIO_METRIC_BOUNDS = OracleAudioMetricBounds(
    mss_max=15.0,
    wmfcc_max=30.0,
    sot_max=0.5,
    rms_min=0.95,
)


def build_oracle_module(num_params: int) -> SurgeFakeOracleModule:
    """Construct a :class:`SurgeFakeOracleModule` matching the surge/fake_oracle config.

    No optimizer step is taken — only ``predict_step`` / eval steps are
    exercised — but the constructor still requires an optimizer factory.

    :param num_params: ``d_out`` for ``FakeOracleNet``; the param-array width.
    :returns: Module ready for direct ``predict_step`` / eval-step calls.
    """
    net = FakeOracleNet(d_out=num_params)
    optimizer = partial(torch.optim.Adam, lr=1e-4)
    return SurgeFakeOracleModule(net=net, optimizer=optimizer, scheduler=None)
