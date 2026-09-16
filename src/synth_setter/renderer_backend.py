"""Shared renderer contracts for launcher configuration and runtime checks.

Interpreter-only (like ``param_spec_name``) so the launcher-pure
``pipeline.schemas.spec`` and the render-worker modules can share definitions
without pulling ``synth_setter.data.vst`` at import time.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type PyFDNExcitation = Literal["chirp", "impulse"]
type RendererBackend = Literal[
    "dawdreamer",
    "faustcpp",
    "faustwasm",
    "pedalboard",
    "pyfdn",
    "surgepy",
    "torchsynth",
]

# ``RenderConfig.plugin_path`` value that selects the in-process backend in
# place of a plugin-bundle path (see ``core.extract_renderer_version``).
TORCHSYNTH_PLUGIN_NAME = "torchsynth"
FAUST_PLUGIN_NAME = "faust"
FAUST_REGISTRY_PREFIX = "registry://faust/"
PYFDN_PLUGIN_NAME = "pyfdn"
SURGEPY_PLUGIN_NAME = "surgepy"

PYFDN_CANONICAL_SOURCE_SHA256 = "5a215ebf9c4f8300774bee0f1e8e6ce5dd4052cb8c422aeeacc16a3d0321e485"
PYFDN_SOURCE_CHANNELS = 1
PYFDN_SOURCE_SAMPLE_RATE_HZ = 44_100
PYFDN_SOURCE_TOTAL_FRAMES = 176_400
# Versioned foreign-audio pinning applied by ``InputAudioPool.take``: librosa soxr_hq
# resample to the renderer rate, mean downmix to mono, then first-N truncate with
# zero-pad tail. Bump when the operations, their order, or the resampler changes so
# contracted digests covering this token retire instead of colliding.
INPUT_AUDIO_ADAPTATION_POLICY = "resample-downmix-pin-v1"


@dataclass(frozen=True)
class FlushBlocks:
    """Silent host blocks processed after preset load, parameter writes, and the note render.

    A block is the host's audio callback. Zero skips that step entirely, including any
    host reset.

    .. attribute :: post_load

       Blocks after the preset loads, before parameter writes.

    .. attribute :: post_param

       Blocks after parameter writes, before the note is scheduled.

    .. attribute :: post_render

       Blocks after the note render, scrubbing voice state.
    """

    post_load: int
    post_param: int
    post_render: int


PEDALBOARD_BLOCK_SIZE = 2048
# Silence Pedalboard processes per flush step so preset state settles deterministically (#489).
PEDALBOARD_FLUSH_SECONDS = 32.0
# Compatibility window measured against Surge identity and Cardinal audio-thread restoration.
DAWDREAMER_PRESET_SETTLE_BLOCKS = 8
DAWDREAMER_FLUSH_BLOCKS = FlushBlocks(
    post_load=DAWDREAMER_PRESET_SETTLE_BLOCKS, post_param=0, post_render=0
)
NO_FLUSH_BLOCKS = FlushBlocks(post_load=0, post_param=0, post_render=0)
FLUSHING_BACKENDS = frozenset({"dawdreamer", "pedalboard"})


def pedalboard_flush_blocks(sample_rate: float) -> FlushBlocks:
    """Return the block counts that cover ``PEDALBOARD_FLUSH_SECONDS`` at ``sample_rate``.

    :param sample_rate: Render sample rate in Hz.
    :returns: Whole-block counts, rounded up, applied to every flush step.
    """
    blocks = math.ceil(PEDALBOARD_FLUSH_SECONDS * sample_rate / PEDALBOARD_BLOCK_SIZE)
    return FlushBlocks(post_load=blocks, post_param=blocks, post_render=blocks)


def default_flush_blocks(renderer_backend: str, sample_rate: float) -> FlushBlocks:
    """Return the flush steps a backend runs when the config leaves them unset.

    :param renderer_backend: Configured renderer backend.
    :param sample_rate: Render sample rate in Hz.
    :returns: Backend defaults; all zeros for backends that never flush.
    """
    if renderer_backend == "pedalboard":
        return pedalboard_flush_blocks(sample_rate)
    if renderer_backend == "dawdreamer":
        return DAWDREAMER_FLUSH_BLOCKS
    return NO_FLUSH_BLOCKS


IN_PROCESS_PLUGIN_NAMES = frozenset(
    {TORCHSYNTH_PLUGIN_NAME, FAUST_PLUGIN_NAME, PYFDN_PLUGIN_NAME, SURGEPY_PLUGIN_NAME}
)


def pyfdn_output_channels(param_spec_name: str) -> int:
    """Return the channel count shared by every pyFDN identity.

    :param param_spec_name: Registered pyFDN param spec name.
    :returns: The mono source channel count.
    """
    del param_spec_name
    return PYFDN_SOURCE_CHANNELS


def missing_render_artifacts(plugin_path: str, plugin_state_path: str) -> tuple[str, ...]:
    """Return the declared render artifacts a renderer would fail to open.

    Mirrors how the hosts resolve their paths: pedalboard's ``VST3Plugin``
    rejects a path that does not exist *before* it scans, so a bundle absent
    from the CWD-relative location is never rescued by the system VST3 search
    paths (``VST3Plugin.installed_plugins`` is a listing helper, not a
    resolution fallback). ``~`` is expanded because ``DawDreamerRenderer``
    expands it.

    :param plugin_path: ``RenderConfig.plugin_path``; in-process backend names and
        registered Faust source URIs are skipped because neither names a bundle on disk.
    :param plugin_state_path: ``RenderConfig.plugin_state_path``; ``""`` when the
        backend takes no preset.
    :returns: The unresolvable paths as declared, bundle before preset.
    """
    logical_source = plugin_path.startswith(FAUST_REGISTRY_PREFIX)
    declared = [] if logical_source or plugin_path in IN_PROCESS_PLUGIN_NAMES else [plugin_path]
    if plugin_state_path:
        declared.append(plugin_state_path)
    return tuple(path for path in declared if not Path(path).expanduser().exists())
