"""Shared constants for the Surge XT CLAP dump tests."""

from __future__ import annotations

# Surge XT 1.3.x exposes exactly this many CLAP params; a different count
# breaks the positional pyname->CLAP index bridge the committed map rests on.
SURGE_XT_CLAP_PARAM_COUNT = 775

# Size of the committed surge_xt map: one entry per SURGE_XT_PARAM_SPEC synth param.
SURGE_XT_MAPPED_PARAM_COUNT = 162
