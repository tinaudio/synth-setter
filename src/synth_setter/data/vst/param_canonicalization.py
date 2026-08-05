"""Pick identifiable point-regression targets from permutation-symmetric blocks.

Registered groups are interchangeable at render level, guarded by a real-render MSS contract in
``tests/data/vst/test_param_canonicalization.py``. Score permutations spectrally, not sample-wise:
the Surge render randomizes phase, so a waveform metric saturates on repeat renders of one row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from synth_setter.data.vst.param_spec import ParamSpec
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName

# Read-only: a mutated entry would silently change every later run's target layout.
# LFO blocks are routed differently and are not render-invariant under permutation — see #1886.
SYMMETRIC_BLOCK_REGISTRY: Mapping[ParamSpecName, tuple[tuple[str, ...], str]] = MappingProxyType(
    {ParamSpecName("surge_simple"): (("a_osc_1_", "a_osc_2_", "a_osc_3_"), "volume")}
)


@dataclass(frozen=True)
class CanonicalBlocks:
    """Encoded-dim layout of one symmetric block group.

    .. attribute :: indices

        Per-block encoded-dim index tuples, congruent across blocks: position
        ``j`` refers to the same param suffix in every block.

    .. attribute :: key_offset

        Within-block position of the sort key; blocks are ordered by
        descending value at this offset.
    """

    indices: tuple[tuple[int, ...], ...]
    key_offset: int


def block_indices_by_prefix(
    spec: ParamSpec, prefixes: Sequence[str], key_suffix: str
) -> CanonicalBlocks:
    """Derive aligned encoded-dim blocks for ``prefixes`` from ``spec``.

    :param spec: Param spec whose synth params are scanned in encoding order.
    :param prefixes: One name prefix per interchangeable block.
    :param key_suffix: Suffix of the within-block param to sort blocks by.
    :returns: Suffix-aligned block indices and the sort-key offset.
    :raises ValueError: If a prefix matches no params, the per-prefix suffix
        sequences differ, or ``key_suffix`` is not among them.
    """
    # Read spans off the spec rather than re-deriving widths, so a parameter
    # type that owns several columns can never be mistaken for a scalar here.
    spans = list(spec.encoded_slices())[: len(spec.synth_params)]
    offsets = {param.name: span.start for param, span in spans}
    widths = {param.name: span.stop - span.start for param, span in spans}

    suffix_orders = []
    for prefix in prefixes:
        matched = [name for name in offsets if name.startswith(prefix)]
        if not matched:
            raise ValueError(f"prefix {prefix!r} matches no synth params in the spec")
        wide = [name for name in matched if widths[name] != 1]
        if wide:
            raise ValueError(f"non-scalar params cannot form canonical blocks: {wide}")
        suffix_orders.append([name.removeprefix(prefix) for name in matched])
    first = suffix_orders[0]
    if any(order != first for order in suffix_orders[1:]):
        raise ValueError(f"blocks {prefixes!r} have mismatched param suffix sequences")
    if key_suffix not in first:
        raise ValueError(f"key_suffix {key_suffix!r} is not a param suffix of {prefixes[0]!r}")

    indices = tuple(tuple(offsets[prefix + suffix] for suffix in first) for prefix in prefixes)
    return CanonicalBlocks(indices=indices, key_offset=first.index(key_suffix))


def canonicalize_blocks(params: np.ndarray, blocks: CanonicalBlocks) -> np.ndarray:
    """Reorder each row's symmetric blocks by descending sort-key value.

    Key ties use the remaining block values as descending lexicographic
    tie-breakers. Dims outside ``blocks`` are untouched.

    :param params: ``(batch, num_params)`` encoded rows; not mutated.
    :param blocks: Block layout to sort within each row.
    :returns: New array with each row's blocks in canonical order.
    """
    out = params.copy()
    block_index_matrix = np.array(blocks.indices)
    gathered = params[:, block_index_matrix]
    tie_break_offsets = [
        offset for offset in range(gathered.shape[2]) if offset != blocks.key_offset
    ]
    priority_offsets = [blocks.key_offset, *tie_break_offsets]
    # Reverse an ascending sort rather than negating: negation wraps on
    # unsigned stored dtypes and would order those blocks backwards.
    sort_keys = tuple(gathered[:, :, offset] for offset in reversed(priority_offsets))
    order = np.lexsort(sort_keys, axis=1)[:, ::-1]
    sorted_blocks = np.take_along_axis(gathered, order[:, :, None], axis=1)
    # Empty batches cannot infer a reshape dimension.
    out[:, block_index_matrix.reshape(-1)] = sorted_blocks.reshape(
        len(params), block_index_matrix.size
    )
    return out


def resolve_canonical_blocks(param_spec_name: ParamSpecName) -> CanonicalBlocks:
    """Resolve the registered symmetric block layout for one spec.

    :param param_spec_name: Registry key naming the spec.
    :returns: The spec's canonical block layout.
    :raises KeyError: If the spec has no registered symmetric blocks.
    """
    if param_spec_name not in SYMMETRIC_BLOCK_REGISTRY:
        raise KeyError(
            f"no symmetric blocks registered for param spec {param_spec_name!r}; "
            f"known: {sorted(SYMMETRIC_BLOCK_REGISTRY)}"
        )
    prefixes, key_suffix = SYMMETRIC_BLOCK_REGISTRY[param_spec_name]
    return block_indices_by_prefix(resolve_param_spec(param_spec_name), prefixes, key_suffix)
