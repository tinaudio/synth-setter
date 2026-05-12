from typing import Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def _hungarian_match(noise: torch.Tensor, params: torch.Tensor, *args):
    if isinstance(noise, np.ndarray):
        noise = torch.from_numpy(noise)
    if isinstance(params, np.ndarray):
        params = torch.from_numpy(params)

    cost = torch.cdist(noise, params)
    cost = cost.numpy()

    with threadpool_limits(limits=1):
        row_ind, col_ind = linear_sum_assignment(cost)

    noise = noise[row_ind]
    params = params[col_ind]

    return_values = [noise, params]
    for arg in args:
        if arg is not None:
            return_values.append(arg[col_ind])
        else:
            return_values.append(None)

    return tuple(return_values)


def concatenate(list_of_arrays: Union[torch.Tensor, np.ndarray]):
    if isinstance(list_of_arrays[0], torch.Tensor):
        return torch.cat(list_of_arrays, dim=0)
    else:
        x = np.concatenate(list_of_arrays, axis=0)
        return torch.from_numpy(x)


def stack(list_of_arrays: Union[torch.Tensor, np.ndarray]):
    if isinstance(list_of_arrays[0], torch.Tensor):
        return torch.stack(list_of_arrays, dim=0)
    else:
        x = np.stack(list_of_arrays, axis=0)
        return torch.from_numpy(x)


def _collate_tuple(batch):
    sins, params, sin_fn = zip(*batch)
    sins = concatenate(sins)
    params = concatenate(params)
    noise = torch.randn_like(params)
    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def _collate_dict(batch):
    # batch is a list of dicts and we want a dict of lists
    params = [d["params"] for d in batch]
    mel_spec = [d["mel_spec"] for d in batch]
    audio = [d["audio"] for d in batch]

    params = stack(params)
    mel_spec = stack(mel_spec)
    if audio[0] is not None:
        audio = stack(audio)
    else:
        audio = None

    noise = torch.randn_like(params)

    return dict(
        params=params,
        noise=noise,
        mel_spec=mel_spec,
        audio=audio,
    )


def regular_collate_fn(batch):
    item = batch[0]
    if isinstance(item, tuple) or isinstance(item, list):
        fn = _collate_tuple
    elif isinstance(item, dict):
        fn = _collate_dict
    else:
        raise NotImplementedError(
            f"Expected tuple or dict for batch type, got {type(item)}"
        )

    return fn(batch)


def _ot_collate_tuple(batch):
    sins, params, noise, sin_fn = _collate_tuple(batch)
    noise, params, sins = _hungarian_match(noise, params, sins)
    return (sins, params, noise, sin_fn)


def _ot_collate_dict(batch):
    batch = _collate_dict(batch)

    noise = batch["noise"]
    params = batch["params"]
    mel_spec = batch["mel_spec"]
    audio = batch["audio"]

    noise, params, mel_spec, audio = _hungarian_match(noise, params, mel_spec, audio)

    return dict(
        params=params,
        noise=noise,
        mel_spec=mel_spec,
        audio=audio,
    )


def ot_collate_fn(batch):
    item = batch[0]
    if isinstance(item, tuple) or isinstance(item, list):
        fn = _ot_collate_tuple
    elif isinstance(item, dict):
        fn = _ot_collate_dict
    else:
        raise NotImplementedError(
            f"Expected tuple or dict for batch type, got {type(item)}"
        )

    return fn(batch)
