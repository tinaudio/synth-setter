"""Shared renderer contracts for launcher configuration and runtime checks.

Interpreter-only (like ``param_spec_name``) so the launcher-pure
``pipeline.schemas.spec`` and the render-worker modules can share definitions
without pulling ``synth_setter.data.vst`` at import time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type PyFDNExcitation = Literal["chirp", "impulse"]
type RendererBackend = Literal[
    "pedalboard",
    "pyfdn",
    "dawdreamer",
    "dawdreamer_faust",
    "surgepy",
    "torchsynth",
]

# ``RenderConfig.plugin_path`` value that selects the in-process backend in
# place of a plugin-bundle path (see ``core.extract_renderer_version``).
TORCHSYNTH_PLUGIN_NAME = "torchsynth"
FAUST_PLUGIN_NAME = "faust"
PYFDN_PLUGIN_NAME = "pyfdn"
SURGEPY_PLUGIN_NAME = "surgepy"

PYFDN_CANONICAL_SOURCE_SHA256 = "5a215ebf9c4f8300774bee0f1e8e6ce5dd4052cb8c422aeeacc16a3d0321e485"
PYFDN_SOURCE_CHANNELS = 1
PYFDN_SOURCE_SAMPLE_RATE_HZ = 44_100
PYFDN_SOURCE_TOTAL_FRAMES = 176_400


@dataclass(frozen=True)
class FlushBlocks:
    """Silent host blocks processed after preset load, parameter writes, and the note render.

    A block is the host's audio callback (2048 samples for both Pedalboard and DawDreamer). Zero
    skips that step entirely, including any host reset.

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


# 32 s of silence at 44.1 kHz rounded up to whole 2048-sample callbacks (#489 preset determinism).
PEDALBOARD_FLUSH_BLOCKS = FlushBlocks(post_load=690, post_param=690, post_render=690)
# Compatibility window measured against Surge identity and Cardinal audio-thread restoration.
DAWDREAMER_PRESET_SETTLE_BLOCKS = 8
DAWDREAMER_FLUSH_BLOCKS = FlushBlocks(
    post_load=DAWDREAMER_PRESET_SETTLE_BLOCKS, post_param=0, post_render=0
)
NO_FLUSH_BLOCKS = FlushBlocks(post_load=0, post_param=0, post_render=0)
FLUSH_BLOCK_DEFAULTS: dict[str, FlushBlocks] = {
    "pedalboard": PEDALBOARD_FLUSH_BLOCKS,
    "dawdreamer": DAWDREAMER_FLUSH_BLOCKS,
}

IN_PROCESS_PLUGIN_NAMES = frozenset(
    {TORCHSYNTH_PLUGIN_NAME, FAUST_PLUGIN_NAME, PYFDN_PLUGIN_NAME, SURGEPY_PLUGIN_NAME}
)


def missing_render_artifacts(plugin_path: str, plugin_state_path: str) -> tuple[str, ...]:
    """Return the declared render artifacts a renderer would fail to open.

    Mirrors how the hosts resolve their paths: pedalboard's ``VST3Plugin``
    rejects a path that does not exist *before* it scans, so a bundle absent
    from the CWD-relative location is never rescued by the system VST3 search
    paths (``VST3Plugin.installed_plugins`` is a listing helper, not a
    resolution fallback). ``~`` is expanded because ``DawDreamerRenderer``
    expands it.

    :param plugin_path: ``RenderConfig.plugin_path``; an in-process backend name
        is skipped, naming no bundle on disk.
    :param plugin_state_path: ``RenderConfig.plugin_state_path``; ``""`` when the
        backend takes no preset.
    :returns: The unresolvable paths as declared, bundle before preset.
    """
    declared = [] if plugin_path in IN_PROCESS_PLUGIN_NAMES else [plugin_path]
    if plugin_state_path:
        declared.append(plugin_state_path)
    return tuple(path for path in declared if not Path(path).expanduser().exists())
