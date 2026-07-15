"""Deterministic dummy-shard writer and a renderer stub for launcher tests.

Both the real-R2 launcher roundtrip (``tests/integration/test_local_launcher_roundtrip.py``)
and the fast fake-R2 orchestrator test (``tests/test_generate_dataset.py``) drive
``cli.generate_dataset`` with the Surge VST3 subprocess replaced by a deterministic
stub that writes a validation-passing Lance shard of the right shape. Centralizing the
stub here keeps the two lanes from drifting: a change to the writer's shard layout
updates both at once. The Lance writer is imported from
``tests.helpers.finalize_shards``, which owns it for the finalize lanes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat
from synth_setter.pipeline.subprocess_stream import check_call_streamed
from tests.helpers.finalize_shards import write_minimal_lance_shard
from tests.helpers.subprocess_args import find_script_index

# rclone passthrough: bound from the pipeline module, so it bypasses the patched
# cli seam (no recursion) yet runs the real streamed runner on the real binary.
_REAL_CHECK_CALL = check_call_streamed


def stub_renderer(spec: DatasetSpec) -> Callable[[list[str]], None]:
    """Return a ``_check_call_streamed`` side effect that writes dummy Lance shards.

    Dispatches on the renderer output path's suffix via ``OutputFormat.from_extension``.
    ``rclone`` invocations fall through to the real binary so the R2 upload, the
    skip-existing probe, and any purge hit the configured remote (real R2, or a
    local-backed fake remote).

    :param spec: Dataset spec the launcher will materialize; threaded into the
        dummy-shard writer so shapes match the validator's expectations.
    :returns: A callable matching ``_check_call_streamed``'s side-effect contract.
    """

    def _side_effect(args: list[str]) -> None:
        if args and args[0] == "rclone":
            _REAL_CHECK_CALL(args)
            return
        script_idx = find_script_index(args)
        output_file = Path(args[script_idx + 1])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fmt = OutputFormat.from_extension(output_file.suffix)
        if fmt is OutputFormat.LANCE:
            write_minimal_lance_shard(output_file, spec)
        else:
            raise AssertionError(
                f"stubbed renderer cannot write output with suffix {output_file.suffix!r}"
            )

    return _side_effect
