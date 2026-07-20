"""Lightning base module for compile-compatible checkpoint loading.

Inherit ``CompiledCheckpointModule`` when ``setup("fit")`` compiles child modules.
"""

import logging
from collections.abc import Iterable, Mapping, MutableMapping
from typing import cast

from lightning import LightningModule
from torch import Tensor

log = logging.getLogger(__name__)

_COMPILED_KEY_PART = "_orig_mod"
_EXTRA_STATE_SUFFIX = "_extra_state"


def _canonical_key(key: str) -> str:
    """Return a key without compile-wrapper path parts.

    :param key: State-dict or metadata key.
    :returns: Key shared across compiled and uncompiled layouts.
    """
    return ".".join(part for part in key.split(".") if part != _COMPILED_KEY_PART)


def _contains_compile_wrapper(key: str) -> bool:
    """Report whether a key contains a compile-wrapper path part.

    :param key: State-dict or metadata key.
    :returns: Whether ``_orig_mod`` is a complete path part.
    """
    return _COMPILED_KEY_PART in key.split(".")


def _compile_wrapper_depth(key: str) -> int:
    """Count compile-wrapper path parts in a key.

    :param key: State-dict or metadata key.
    :returns: Number of ``_orig_mod`` path parts.
    """
    return key.split(".").count(_COMPILED_KEY_PART)


def _grouped_keys(keys: Iterable[str]) -> dict[str, list[str]]:
    """Group keys by their compile-independent representation.

    :param keys: State-dict or metadata keys.
    :returns: Keys indexed by their canonical representation.
    """
    grouped: dict[str, list[str]] = {}
    for key in keys:
        grouped.setdefault(_canonical_key(key), []).append(key)
    return grouped


def _state_signature(
    key: str,
    value: object,
) -> tuple[str, tuple[int, ...] | None] | None:
    """Return the load-compatibility signature for one state value.

    :param key: State-dict key.
    :param value: State-dict value.
    :returns: Compatibility signature, or ``None`` for unsupported values.
    """
    if key.rsplit(".", maxsplit=1)[-1] == _EXTRA_STATE_SUFFIX:
        return (_EXTRA_STATE_SUFFIX, None)
    if isinstance(value, Tensor):
        return ("tensor", tuple(value.shape))
    return None


def _unique_compatible_pairs(
    source_keys: list[str],
    target_keys: list[str],
    source_state: Mapping[str, object],
    target_state: Mapping[str, object],
) -> list[tuple[str, str]] | None:
    """Return the sole value-compatible perfect matching between two key groups.

    :param source_keys: Canonically equivalent checkpoint keys.
    :param target_keys: Canonically equivalent live keys.
    :param source_state: Checkpoint values indexed by key.
    :param target_state: Live values indexed by key.
    :returns: Unique matching, or ``None`` when no safe matching is provable.
    """
    source_by_signature = {_state_signature(key, source_state[key]): key for key in source_keys}
    target_by_signature = {_state_signature(key, target_state[key]): key for key in target_keys}
    if (
        None in source_by_signature
        or None in target_by_signature
        or len(source_by_signature) != len(source_keys)
        or len(target_by_signature) != len(target_keys)
        or source_by_signature.keys() != target_by_signature.keys()
    ):
        return None
    return [
        (source_by_signature[signature], target_by_signature[signature])
        for signature in source_by_signature
    ]


def _state_replacements(
    source_state: Mapping[str, object],
    live_state: Mapping[str, object],
) -> tuple[list[tuple[str, str]], set[str]]:
    """Find unambiguous compile-wrapper key replacements.

    :param source_state: Checkpoint state dict.
    :param live_state: Current module state dict.
    :returns: Key replacements and their canonical groups.
    """
    live_groups = _grouped_keys(live_state)
    replacements: list[tuple[str, str]] = []
    changed_groups: set[str] = set()
    for canonical_key, source_keys in _grouped_keys(source_state).items():
        target_keys = live_groups.get(canonical_key)
        if target_keys is None or set(source_keys) == set(target_keys):
            continue
        if len(source_keys) != len(target_keys) or not any(
            _contains_compile_wrapper(key) for key in source_keys + target_keys
        ):
            continue
        pairs = _unique_compatible_pairs(
            source_keys,
            target_keys,
            source_state,
            live_state,
        )
        if pairs is not None:
            replacements.extend(pairs)
            changed_groups.add(canonical_key)
    return replacements, changed_groups


def _apply_state_replacements(
    state_dict: MutableMapping[str, object],
    replacements: list[tuple[str, str]],
) -> None:
    """Apply key replacements without overwriting colliding source slots.

    :param state_dict: Checkpoint state dict mutated in place.
    :param replacements: Source-to-target key pairs.
    """
    replacement_values = {
        target_key: state_dict[source_key] for source_key, target_key in replacements
    }
    for source_key, target_key in replacements:
        if source_key != target_key:
            state_dict.pop(source_key)
    state_dict.update(replacement_values)


def _normalize_metadata(
    metadata: MutableMapping[str, dict[str, object]],
    live_metadata: Mapping[str, dict[str, object]],
    changed_groups: set[str],
) -> None:
    """Align underlying-module metadata with changed state-key groups.

    :param metadata: Checkpoint metadata mutated in place.
    :param live_metadata: Current module metadata layout.
    :param changed_groups: Canonical state-key groups that were replaced.
    """
    target_metadata = _grouped_keys(live_metadata)
    for canonical_key, source_keys in _grouped_keys(metadata).items():
        live_candidates = target_metadata.get(canonical_key)
        if (
            live_candidates is None
            or set(source_keys) == set(live_candidates)
            or not any(
                key == canonical_key or key.startswith(f"{canonical_key}.")
                for key in changed_groups
            )
        ):
            continue
        pair_count = min(len(source_keys), len(live_candidates))
        ordered_sources = sorted(
            source_keys,
            key=lambda key: (_compile_wrapper_depth(key), key),
        )[-pair_count:]
        ordered_targets = sorted(
            live_candidates,
            key=lambda key: (_compile_wrapper_depth(key), key),
        )[-pair_count:]
        values = [metadata[key] for key in ordered_sources]
        for key in source_keys:
            metadata.pop(key)
        metadata.update(zip(ordered_targets, values, strict=True))


class CompiledCheckpointModule(LightningModule):
    """Normalize checkpoint keys to the module's current compile state."""

    def on_load_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        """Align checkpoint keys with compiled or uncompiled live parameters.

        :param checkpoint: Lightning checkpoint mutated before weight loading.
        """
        state_dict = cast(MutableMapping[str, object], checkpoint["state_dict"])
        live_state_dict = cast(Mapping[str, object], self.state_dict())
        replacements, changed_groups = _state_replacements(state_dict, live_state_dict)
        _apply_state_replacements(state_dict, replacements)

        metadata = cast(
            MutableMapping[str, dict[str, object]] | None,
            getattr(state_dict, "_metadata", None),
        )
        live_metadata = cast(
            Mapping[str, dict[str, object]],
            getattr(live_state_dict, "_metadata", {}),
        )
        if replacements and metadata:
            _normalize_metadata(metadata, live_metadata, changed_groups)

        if replacements:
            log.info("Normalized %d compiled checkpoint keys", len(replacements))
