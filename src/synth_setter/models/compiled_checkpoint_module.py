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


class CompiledCheckpointModule(LightningModule):
    """Normalize checkpoint keys to the module's current compile state."""

    def on_load_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        """Align checkpoint keys with compiled or uncompiled live parameters.

        :param checkpoint: Lightning checkpoint mutated before weight loading.
        """
        state_dict = cast(MutableMapping[str, Tensor], checkpoint["state_dict"])
        live_state_dict = self.state_dict()
        source_groups = _grouped_keys(state_dict)
        live_groups = _grouped_keys(live_state_dict)
        replacements: list[tuple[str, str]] = []
        changed_groups: set[str] = set()
        for canonical_key, source_keys in source_groups.items():
            target_keys = live_groups.get(canonical_key)
            if target_keys is None or set(source_keys) == set(target_keys):
                continue
            if len(source_keys) != len(target_keys) or not any(
                _contains_compile_wrapper(key) for key in source_keys + target_keys
            ):
                continue
            ordered_sources = sorted(
                source_keys,
                key=lambda key: (_compile_wrapper_depth(key), key),
            )
            ordered_targets = sorted(
                target_keys,
                key=lambda key: (_compile_wrapper_depth(key), key),
            )
            pairs = list(zip(ordered_sources, ordered_targets, strict=True))
            if any(
                state_dict[source_key].shape != live_state_dict[target_key].shape
                for source_key, target_key in pairs
            ):
                continue
            replacements.extend(pairs)
            changed_groups.add(canonical_key)

        replacement_values = {
            target_key: state_dict[source_key] for source_key, target_key in replacements
        }
        for source_key, target_key in replacements:
            if source_key != target_key:
                state_dict.pop(source_key)
        state_dict.update(replacement_values)

        metadata = cast(
            MutableMapping[str, dict[str, object]] | None,
            getattr(state_dict, "_metadata", None),
        )
        live_metadata = cast(
            Mapping[str, dict[str, object]],
            getattr(live_state_dict, "_metadata", {}),
        )
        if replacements and metadata:
            source_metadata = _grouped_keys(metadata)
            target_metadata = _grouped_keys(live_metadata)

            for canonical_key, source_keys in source_metadata.items():
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

        if replacements:
            log.info("Normalized %d compiled checkpoint keys", len(replacements))
