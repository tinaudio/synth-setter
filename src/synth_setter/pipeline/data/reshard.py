from pathlib import Path

import click
import h5py
import numpy as np


@click.command()
@click.argument("dataset_root", type=str)
@click.option("--train-samples", "-t", type=int, default=200)
@click.option("--val-samples", "-v", type=int, default=4)
@click.option("--test-samples", "-e", type=int, default=1)
def main(
    dataset_root: str,
    train_samples: int = 200,
    val_samples: int = 4,
    test_samples: int = 1,
):
    dataset_root = Path(dataset_root)
    files = dataset_root.glob("shard-*.h5")
    files = sorted(list(files))
    assert len(files) == train_samples + val_samples + test_samples

    splits = {
        "train": files[:train_samples],
        "val": files[train_samples : train_samples + val_samples],
        "test": files[train_samples + val_samples :],
    }

    for split, files in splits.items():
        print(split)
        split_len = len(files) * 10_000

        with h5py.File(files[0], "r") as f:
            audio_shape = f["audio"].shape[1:]
            mel_shape = f["mel_spec"].shape[1:]
            param_shape = f["param_array"].shape[1:]

        vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
        vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
        vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

        for i, file in enumerate(files):
            vs_audio = h5py.VirtualSource(
                file, "audio", dtype=np.float32, shape=(10_000, *audio_shape)
            )
            vs_mel = h5py.VirtualSource(
                file, "mel_spec", dtype=np.float32, shape=(10_000, *mel_shape)
            )
            vs_param = h5py.VirtualSource(
                file, "param_array", dtype=np.float32, shape=(10_000, *param_shape)
            )

            range_start = i * 10_000
            range_end = (i + 1) * 10_000

            print(range_start, range_end)
            vl_audio[range_start:range_end, :, :] = vs_audio
            vl_mel[range_start:range_end, :, :, :] = vs_mel
            vl_param[range_start:range_end, :] = vs_param

        split_file = str(dataset_root / f"{split}.h5")
        with h5py.File(split_file, "w") as f:
            f.create_virtual_dataset("audio", vl_audio)
            f.create_virtual_dataset("mel_spec", vl_mel)
            f.create_virtual_dataset("param_array", vl_param)


if __name__ == "__main__":
    main()
