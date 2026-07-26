"""Text renderings of a param spec, used as conditioning prompts.

Normalizers take the whole encoded batch so a values-aware strategy can be added without changing
callers, even though the only strategy here ignores values.
"""

from collections.abc import Callable

import numpy as np

from synth_setter.data.vst.param_spec import ParamSpec

type ParamTextNormalizer = Callable[[ParamSpec, np.ndarray], list[str]]

DEFAULT_PARAM_TEXT_NORMALIZER: str = "param_names"


def param_names_normalizer(spec: ParamSpec, rows: np.ndarray) -> list[str]:
    """Render every row as the spec's comma-separated parameter names.

    Names run in encode order (synth then note), so the caption enumerates the
    same space as an encoded row. Values are ignored, making the caption
    identical for every row of one spec.

    :param spec: Param spec whose names describe the encoded rows.
    :param rows: Encoded ``(B, encoded_width)`` batch, read only for its length.
    :returns: One caption per row in ``rows``.
    """
    return [", ".join(spec.names)] * len(rows)


PARAM_TEXT_NORMALIZERS: dict[str, ParamTextNormalizer] = {
    DEFAULT_PARAM_TEXT_NORMALIZER: param_names_normalizer,
}


def resolve_param_text_normalizer(name: str) -> ParamTextNormalizer:
    """Look up a registered text normalizer by name.

    :param name: Registered normalizer name.
    :returns: The matching normalizer.
    :raises KeyError: No normalizer is registered under ``name``.
    """
    if name not in PARAM_TEXT_NORMALIZERS:
        known = ", ".join(sorted(PARAM_TEXT_NORMALIZERS))
        raise KeyError(f"unknown param text normalizer {name!r}; known normalizers: {known}")
    return PARAM_TEXT_NORMALIZERS[name]
