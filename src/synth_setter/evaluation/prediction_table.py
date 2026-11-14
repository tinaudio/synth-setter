"""Lance table of eval predictions: one render-ready row per predicted sample.

Each row pairs the raw model output with the quantized row a renderer plays and,
when the split carries ground truth, the target params — everything a render
worker needs to produce a prediction/target audio pair without the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

from synth_setter.data.vst.param_spec import ParamSpec, spec_quantize_model_output
from synth_setter.pipeline.data.lance_shard import LANCE_DATA_STORAGE_VERSION, tensor_array

ROW_ID_FIELD = "row_id"
PRED_PARAMS_RAW_FIELD = "pred_params_raw"
PRED_PARAMS_EFFECTIVE_FIELD = "pred_params_effective"
TARGET_PARAMS_FIELD = "target_params"
PREDICTION_TABLE_NAME = "predictions.lance"

_PARAM_DTYPE = np.dtype(np.float32)


def prediction_rows(
    predictions: np.ndarray, target_params: np.ndarray | None, spec: ParamSpec
) -> pa.Table:
    """Build prediction-table rows for one batch of model output.

    A wrong-width or non-finite prediction, or targets that do not pair up with the
    predictions row for row, surface as ``ValueError`` from the encoders.

    :param predictions: Model-space predictions shaped ``(rows, spec.encoded_width)``.
    :param target_params: Model-space ground truth with the same shape, or ``None``
        for corpora without a ground-truth patch; the target column is then omitted.
    :param spec: Parameter spec the model was trained against.
    :returns: Arrow table of raw and effective predictions, plus targets when given.
    """
    raw = np.asarray(predictions, dtype=_PARAM_DTYPE)
    width = (spec.encoded_width,)
    effective = np.stack([spec_quantize_model_output(row, spec) for row in raw])
    columns: dict[str, pa.Array] = {
        PRED_PARAMS_RAW_FIELD: tensor_array(raw, _PARAM_DTYPE, width),
        PRED_PARAMS_EFFECTIVE_FIELD: tensor_array(effective, _PARAM_DTYPE, width),
    }
    if target_params is not None:
        columns[TARGET_PARAMS_FIELD] = tensor_array(target_params, _PARAM_DTYPE, width)
    return pa.table(columns)


def write_prediction_table(
    uri: Path | str,
    batches: Sequence[pa.Table],
    *,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Write a predict pass as one table, replacing any table already at ``uri``.

    :param uri: Table location (local path or object-store URI).
    :param batches: Non-empty :func:`prediction_rows` tables in predict order, all with
        or all without targets; ``row_id`` numbers their rows from 0 in that order.
    :param storage_options: Object-store config for a cloud ``uri``; ``None`` locally.
    """
    rows = pa.concat_tables(batches)
    row_ids = pa.array(range(rows.num_rows), pa.int64())
    lance.write_dataset(
        rows.add_column(0, ROW_ID_FIELD, row_ids),
        str(uri),
        mode="overwrite",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
    )
