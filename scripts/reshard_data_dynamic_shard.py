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

    for split, split_files in splits.items():
        print(split)

        shard_sizes = []
        for f in split_files:
            with h5py.File(f, "r") as h:
                shard_sizes.append(h["audio"].shape[0])
        split_len = sum(shard_sizes)

        with h5py.File(split_files[0], "r") as f:
            audio_shape = f["audio"].shape[1:]
            mel_shape = f["mel_spec"].shape[1:]
            param_shape = f["param_array"].shape[1:]

        vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
        vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
        vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

        offset = 0
        for file, size in zip(split_files, shard_sizes):
            vs_audio = h5py.VirtualSource(
                file, "audio", dtype=np.float32, shape=(size, *audio_shape)
            )
            vs_mel = h5py.VirtualSource(
                file, "mel_spec", dtype=np.float32, shape=(size, *mel_shape)
            )
            vs_param = h5py.VirtualSource(
                file, "param_array", dtype=np.float32, shape=(size, *param_shape)
            )

            print(offset, offset + size)
            vl_audio[offset : offset + size] = vs_audio
            vl_mel[offset : offset + size] = vs_mel
            vl_param[offset : offset + size] = vs_param
            offset += size

        split_file = str(dataset_root / f"{split}.h5")
        with h5py.File(split_file, "w") as f:
            f.create_virtual_dataset("audio", vl_audio)
            f.create_virtual_dataset("mel_spec", vl_mel)
            f.create_virtual_dataset("param_array", vl_param)


if __name__ == "__main__":
    main()
