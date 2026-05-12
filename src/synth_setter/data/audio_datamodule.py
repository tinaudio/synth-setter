from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
from lightning import LightningDataModule
from pedalboard.io import AudioFile


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Values hardcoded to be roughly like those used by the audio spectrogram transformer.

    i.e. 100 frames per second, 128 mels, ~25ms window, hamming window.
    """

    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    window = "hamming"

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        segment_length_seconds: float = 4.0,
        reference_stats_file: Optional[str] = None,
        amp_scale: float = 0.5,
        sample_rate: float = 44100.0,
    ):
        self.segment_length_seconds = segment_length_seconds

        self.root = Path(root)
        self.files = list(self.root.glob("*.wav"))

        self.amp_scale = amp_scale
        self.sample_rate = sample_rate

        self._load_stats(reference_stats_file)

    def _load_stats(self, reference_stats_file: Optional[str]):
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
    def get_stats_file_path(root: Union[str, Path]) -> Path:
        data_dir = Path(root)
        return data_dir / "stats.npz"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]

        length_seconds = max(self.segment_length_seconds - 0.05, 0.0)

        with AudioFile(str(file), "r").resampled_to(self.sample_rate) as f:
            sample_rate = f.samplerate
            num_frames = int(sample_rate * length_seconds)
            audio = f.read(num_frames)

        channels, _ = audio.shape
        if channels == 1:
            audio = np.concatenate([audio, audio], axis=0)
        elif channels > 2:
            raise ValueError(
                f"Audio must have two or fewer channels. Found {channels}."
            )

        start_samples = int(0.05 * sample_rate)
        target_samples = int(sample_rate * self.segment_length_seconds)
        audio = np.pad(audio, [(0, 0), (start_samples, 0)], mode="constant")

        if audio.shape[1] > target_samples:
            audio = audio[:, :num_frames]

        elif audio.shape[1] < target_samples:
            audio = np.pad(
                audio, [(0, 0), (0, target_samples - audio.shape[1])], mode="constant"
            )

        audio = audio * self.amp_scale

        spec = make_spectrogram(audio, sample_rate)
        if self.mean is not None:
            spec = (spec - self.mean) / self.std

        audio = torch.from_numpy(audio).to(dtype=torch.float32)
        spec = torch.from_numpy(spec).to(dtype=torch.float32)

        return {
            "audio": audio,
            "mel_spec": spec,
        }


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        segment_length_seconds: float = 4.0,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        stats_file: Optional[str] = None,
    ):
        super().__init__()

        self.root = root
        self.segment_length_seconds = segment_length_seconds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.stats_file = stats_file

    def setup(self, stage: Optional[str] = None):
        self.predict_dataset = AudioFolderDataset(
            self.root, self.segment_length_seconds, self.stats_file
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
