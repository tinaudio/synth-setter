"""Test helpers for inspecting subprocess argv lists produced by the launcher.

Extracted so the two test files that need this — the unit-tier
``tests/pipeline/entrypoints/test_generate_dataset_unit.py`` mocks and the
launcher-roundtrip ``tests/integration/test_local_launcher_roundtrip.py`` —
share one source of truth; if the renderer script name or the launcher's
argv layout ever changes, both lanes pick the change up automatically.
"""

from __future__ import annotations


def find_script_index(args: list[str]) -> int:
    """Locate ``generate_vst_dataset.py`` in subprocess args.

    The argv layout depends on platform — ``[wrapper, python, script, output, ...]``
    on Linux versus ``[python, script, output, ...]`` elsewhere — so callers
    locate the script by name rather than fixed index.

    :param args: argv list passed to a patched ``_check_call_streamed``.
    :returns: Index of the entry that ends with ``generate_vst_dataset.py``.
    :raises AssertionError: No entry in ``args`` ends with the script name.
    """
    for i, a in enumerate(args):
        if a.endswith("generate_vst_dataset.py"):
            return i
    raise AssertionError(f"generate_vst_dataset.py not found in subprocess args: {args}")
