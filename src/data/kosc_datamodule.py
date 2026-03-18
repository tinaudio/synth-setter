import sys
from functools import partial
from typing import Optional, Tuple, Union

import torch
from lightning import LightningDataModule

from src.data.ot import ot_collate_fn, regular_collate_fn
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def polyblep_sawtooth(frequency: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    dt = frequency[..., None] / (2 * torch.pi)
    phase = dt * n
    phase.fmod_(1.0)

    sawtooth = phase.mul(2.0).sub_(1.0)

    dt_safe = dt.clamp_min(1e-6)
    correction_low = (phase / dt_safe - 1).square()
    correction_high = -(((phase - 1) / dt_safe + 1).square())

    correction_low = torch.where(phase < dt, correction_low, 0.0)
    correction_high = torch.where(phase > 1 - dt, correction_high, 0.0)

    sawtooth.add_(correction_low).add_(correction_high)

    return sawtooth


# @torch.jit.script
def polyblep_square(frequency: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    dt = frequency[..., None] / (2 * torch.pi)
    phase = dt * n
    phase.fmod_(1.0)

    shifted_phase = phase.sub(0.5) % 1.0
    square = torch.where(phase > 0.5, 1.0, -1.0)

    low_blep = phase < dt
    mid_low_blep = (phase > 0.5) & (phase < 0.5 + dt)
    mid_high_blep = (phase > 0.5 - dt) & (phase < 0.5)
    high_blep = phase > 1.0 - dt

    dt_safe = dt.clamp_min(1e-6)
    correction_low = (phase / dt_safe - 1).square_()
    correction_mid_low = -((shifted_phase / dt_safe - 1).square_())
    correction_mid_high = ((shifted_phase - 1) / dt_safe + 1).square_()
    correction_high = -((phase - 1) / dt_safe + 1).square_()

    correction_low = torch.where(low_blep, correction_low, 0.0)
    correction_mid_low = torch.where(mid_low_blep, correction_mid_low, 0.0)
    correction_mid_high = torch.where(mid_high_blep, correction_mid_high, 0.0)
    correction_high = torch.where(high_blep, correction_high, 0.0)

    square.add_(correction_low).add_(correction_mid_low).add_(correction_mid_high).add_(
        correction_high
    )

    return square


def make_sig(params: torch.Tensor, length: int, break_symmetry: bool = False):
    params = params.clamp(-1.0, 1.0)
    freqs, amps, waveform = params.chunk(3, dim=-1)

    freqs = (freqs + 1.0) / 2.0
    if break_symmetry:
        k = freqs.shape[-1]
        shift = torch.arange(k, device=freqs.device) / k
        freqs = shift + freqs / k

    min_freq = 2 * torch.pi * 20.0 / 44100.0
    max_freq = 2 * torch.pi * 4000.0 / 44100.0
    freqs = min_freq + (max_freq - min_freq) * freqs

    amps = (amps + 1.0) / 2.0

    n = torch.arange(length, device=freqs.device)
    phi = freqs[..., None] * n
    sins = -torch.sin(phi)

    squares = polyblep_square(freqs, n)
    saws = polyblep_sawtooth(freqs, n)
    # squares = torch.zeros_like(sins)
    # saws = torch.zeros_like(sins)

    # -1 sin, 0 square, 1 sawtooth
    waveform = waveform[..., None]
    sin_amt = torch.clamp(-1 * waveform, 0.0, 1.0)
    square_amt = 1 - torch.abs(waveform)
    saw_amt = torch.clamp(waveform, 0.0, 1.0)
    x = sin_amt * sins + square_amt * squares + saw_amt * saws

    x = x * amps[..., None]

    return x.mean(dim=-2)


class KOscDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        k: int,
        signal_length: int,
        num_samples: int,
        sort_frequencies: bool,
        break_symmetry: bool,
        is_test: bool,
        seed: int,
        debug_num_samples: Optional[int] = None,
    ):
        self.k = k
        self.signal_length = signal_length

        self.sort_frequencies = sort_frequencies
        self.break_symmetry = break_symmetry

        self.num_samples = num_samples

        self.seed = seed
        self.generator = torch.Generator(device=torch.device("cpu"))

        self.is_test = is_test

        self.debug_num_samples = debug_num_samples

        self._init_dataset()

    def _init_dataset(self):
        self.generator.manual_seed(self.seed)
        # freqs, amps = self._sample_parameters()
        # # freqs.share_memory_()
        # # amps.share_memory_()
        # log.info("Done!")
        # self.freqs = freqs
        # self.amps = amps

    def _sample_parameters(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.generator.manual_seed(seed)

        params = torch.empty(1, 3 * self.k, device=torch.device("cpu"))
        params.uniform_(-1.0, 1.0, generator=self.generator)
        if self.sort_frequencies:
            freqs, amps, waveform = params.chunk(3, dim=-1)
            freqs, _ = torch.sort(freqs, dim=-1)
            params = torch.cat((freqs, amps, waveform), dim=-1)

        return params

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # modulo max int to avoid overflows
        seed = (self.seed * idx) % sys.maxsize

        if self.debug_num_samples is not None:
            seed = seed % self.debug_num_samples

        params = self._sample_parameters(seed)
        render_fn = partial(
            make_sig, length=self.signal_length, break_symmetry=self.break_symmetry
        )
        sig = render_fn(params)
        return (sig, params, render_fn)


class KOscDataModule(LightningDataModule):
    """K-Osc is a simple synthetic synthesiser parameter estimation task designed to elicit
    problematic behaviour in response to permutation invariant labels.

    Each item consists of a signal containing a mixture of sinusoids, and the amplitude and
    frequency parameters used to generate the sinusoids.
    """

    def __init__(
        self,
        k: int,
        signal_length: int = 1024,
        sort_frequencies: bool = False,
        break_symmetry: bool = False,
        train_val_test_sizes: Tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: Tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 1024,
        ot: bool = False,
        num_workers: int = 0,
        debug_num_samples: Optional[int] = None,
    ):
        super().__init__()

        # signal
        self.k = k
        self.signal_length = signal_length
        self.sort_frequencies = sort_frequencies
        self.break_symmetry = break_symmetry

        # dataset
        self.train_size, self.val_size, self.test_size = train_val_test_sizes
        self.train_seed, self.val_seed, self.test_seed = train_val_test_seeds

        # dataloader
        self.batch_size = batch_size

        self.device = None
        self.num_workers = num_workers

        self.ot = ot
        self.debug_num_samples =debug_num_samples

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            train_ds = KOscDataset(
                self.k,
                self.signal_length,
                self.train_size,
                self.sort_frequencies,
                self.break_symmetry,
                False,
                self.train_seed,
                self.debug_num_samples,
            )
            val_ds = KOscDataset(
                self.k,
                self.signal_length,
                self.val_size,
                self.sort_frequencies,
                self.break_symmetry,
                False,
                self.val_seed,
                self.debug_num_samples,
            )
            self.train = torch.utils.data.DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=ot_collate_fn if self.ot else regular_collate_fn,
                num_workers=self.num_workers,
            )
            self.val = torch.utils.data.DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=regular_collate_fn,
                num_workers=self.num_workers,
            )
        else:
            test_ds = KOscDataset(
                self.k,
                self.signal_length,
                self.test_size,
                self.sort_frequencies,
                self.break_symmetry,
                True,
                self.test_seed,
                self.debug_num_samples,
            )
            self.test = torch.utils.data.DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=regular_collate_fn,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: Optional[str] = None):
        pass


if __name__ == "__main__":
    dm = KOscDataModule(k=4)
    dm.setup("fit")
    for x, y in dm.train:
        print(x.shape, y.shape)
        break
