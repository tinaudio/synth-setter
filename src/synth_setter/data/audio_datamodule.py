from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from lightning import LightningDataModule
from pedalboard.io import AudioFile

from synth_setter.data.vst.shapes import make_spectrogram


@jaxtyped(typechecker=beartype)
def load_audio_file_to_grid(
    path: str | Path,
    *,
    segment_length_seconds: float = 4.0,
    leading_padding_seconds: float = 0.05,
    amp_scale: float = 0.5,
    sample_rate: float | int = 44100.0,
) -> Float[torch.Tensor, "2 samples"]:
    """Load a mono/stereo file onto a fixed stereo model grid.

    :param path: Source audio accepted by Pedalboard.
    :param segment_length_seconds: Output duration in seconds.
    :param leading_padding_seconds: Silence prepended before source audio.
    :param amp_scale: Amplitude multiplier applied after fitting.
    :param sample_rate: Resampling and output sample rate in hertz.
    :returns: Float32 stereo waveform on the requested sample grid.
    :raises ValueError: The source has more than two channels.
    """
    resolved_sample_rate = float(sample_rate)
    source_length_seconds = max(segment_length_seconds - leading_padding_seconds, 0.0)
    with AudioFile(str(path), "r").resampled_to(resolved_sample_rate) as audio_file:
        audio = audio_file.read(int(resolved_sample_rate * source_length_seconds))

    channels = audio.shape[0]
    if channels == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif channels != 2:
        raise ValueError(f"Audio must have two or fewer channels. Found {channels}.")

    leading_samples = int(leading_padding_seconds * resolved_sample_rate)
    target_samples = int(resolved_sample_rate * segment_length_seconds)
    audio = np.pad(audio, ((0, 0), (leading_samples, 0)), mode="constant")
    audio = audio[:, :target_samples]
    if audio.shape[1] < target_samples:
        audio = np.pad(audio, ((0, 0), (0, target_samples - audio.shape[1])), mode="constant")
    audio = audio * amp_scale
    return torch.from_numpy(audio).to(dtype=torch.float32)


class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        segment_length_seconds: float = 4.0,
        reference_stats_file: str | None = None,
        amp_scale: float = 0.5,
        sample_rate: float = 44100.0,
        files: list[Path] | None = None,
    ):
        self.segment_length_seconds = segment_length_seconds

        self.root = Path(root)
        # An explicit file list skips the folder glob — single-capture callers
        # (cli/predict_capture.py) must not pay a scan of the whole capture dir.
        self.files = list(files) if files is not None else sorted(self.root.glob("*.wav"))

        self.amp_scale = amp_scale
        self.sample_rate = sample_rate

        self._load_stats(reference_stats_file)

    def _load_stats(self, reference_stats_file: str | None):
        if reference_stats_file is None:
            self.mean = None
            self.std = None
            return

        with np.load(reference_stats_file) as stats:
            self.mean = stats["mean"]
            self.std = stats["std"]

        # TODO: think this through better --- how do we rescale after prediction?

        # dataset_stats_file = AudioFolderDataset.get_stats_file_path(self.root)
        # if not dataset_stats_file.exists():
        #     return
        #
        # dataset_stats = np.load(dataset_stats_file)
        # dataset_mean = dataset_stats["mean"]
        # dataset_std = dataset_stats["std"]
        #
        # D = self.mean - dataset_mean
        # beta = np.mean(D)
        #
        # frob_inner = np.sum(self.std * dataset_std)
        # frob_ood = np.linalg.norm(dataset_std) ** 2
        # gamma = frob_inner / frob_ood
        #
        # self.mean = self.mean - beta
        # self.std = self.std / gamma

    @staticmethod
    def get_stats_file_path(root: str | Path) -> Path:
        data_dir = Path(root)
        return data_dir / "stats.npz"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        audio = load_audio_file_to_grid(
            self.files[idx],
            segment_length_seconds=self.segment_length_seconds,
            amp_scale=self.amp_scale,
            sample_rate=self.sample_rate,
        )

        spec = make_spectrogram(audio.numpy(), self.sample_rate)
        if self.mean is not None:
            spec = (spec - self.mean) / self.std

        spec = torch.from_numpy(spec).to(dtype=torch.float32)

        return {
            "audio": audio,
            "mel": spec,
        }


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        segment_length_seconds: float = 4.0,
        batch_size: int = 32,
        num_workers: int = 0,
        stats_file: str | None = None,
    ):
        super().__init__()

        self.root = root
        self.segment_length_seconds = segment_length_seconds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stats_file = stats_file

    def setup(self, stage: str | None = None):
        self.predict_dataset = AudioFolderDataset(
            self.root, self.segment_length_seconds, self.stats_file
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
