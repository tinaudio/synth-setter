from functools import partial
from typing import Literal, Optional, Tuple, Union

import torch
from lightning import LightningDataModule

from src.data.ot import ot_collate_fn, regular_collate_fn


def _sample_freqs(
    k: int,
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.empty(num_samples, k, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )


def _sample_amplitudes(
    k: int,
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.empty(num_samples, k, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )


def _scale_freqs_and_amps_fm(freqs: torch.Tensor, amps: torch.Tensor):
    # max freq = torch.pi / 10
    freqs = (torch.pi / 4.0) * (freqs + 1.0) / 2.0
    amps = (amps + 1.0) / 2.0
    return freqs, amps


def _sample_freqs_symmetry_broken(
    k: int,
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample frequencies such that each sinusoidal component has frequency drawn from disjoint
    intervals."""
    freqs = _sample_freqs(k, num_samples, device, generator) / k
    shift = 2.0 * torch.arange(k, device=device) / k
    return freqs + shift[None, :]


def fm_conditional_symmetry(
    params: torch.Tensor, length: int, break_symmetry: bool = False
):
    """An FM synthesiser with algorithm (M1->C1) + C2.

    param layout: [m1, c2, c1]
    """
    freqs, amps = params.chunk(2, dim=-1)

    assert freqs.shape[-1] == 3
    assert amps.shape[-1] == 3

    # mod indices should go higher than amplitudes
    amps = amps.clone()
    amps[..., 0] *= 2
    freqs, amps = _scale_freqs_and_amps_fm(freqs, amps)

    if break_symmetry:
        freqs[:, 0] = freqs[:, 0] / 2
        freqs[:, 1] = (freqs[:, 1] + torch.pi) / 2

    n = torch.arange(length, device=freqs.device)
    phi_m1c2 = freqs[..., :2, None] * n
    x_m1c2 = torch.sin(phi_m1c2)
    x_m1c2 = x_m1c2 * amps[..., :2, None]
    x_m1, x_c2 = x_m1c2.chunk(2, dim=-2)

    phi_c1 = freqs[..., 2:3, None] * n + x_m1
    x_c1 = torch.sin(phi_c1)
    x_c1 = x_c1 * amps[..., 2:3, None]

    return x_c1.squeeze(-2) + x_c2.squeeze(-2)


def _sample_params_conditional_symmetry(
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator],
):
    amplitudes = _sample_amplitudes(3, num_samples, device, generator)
    freqs = _sample_freqs(3, num_samples, device, generator)
    return freqs, amplitudes


def fm_mixed_symmetry(params: torch.Tensor, length: int, break_symmetry: bool = False):
    """An FM synthesiser with algorithm (M1->C1) + (M2->C2)
    layout: [m1, m2, c1, c2]
    """
    freqs, amps = params.chunk(2, dim=-1)
    assert freqs.shape[-1] == 4
    assert amps.shape[-1] == 4
    n = torch.arange(length, device=freqs.device)

    # mod indices should go higher than amplitudes
    amps = amps.clone()
    amps[..., 0:2] *= 2
    freqs, amps = _scale_freqs_and_amps_fm(freqs, amps)

    if break_symmetry:
        freqs[:, 0] = freqs[:, 0] / 2
        freqs[:, 1] = (freqs[:, 1] + torch.pi) / 2
        freqs[:, 2] = freqs[:, 2] / 2
        freqs[:, 3] = (freqs[:, 3] + torch.pi) / 2

    phi_m1m2 = freqs[..., :2, None] * n
    x_m1m2 = torch.sin(phi_m1m2)
    x_m1m2 = x_m1m2 * amps[..., :2, None]

    phi_c1c2 = freqs[..., 2:4, None] * n + x_m1m2
    x_c1c2 = torch.sin(phi_c1c2)
    x_c1c2 = x_c1c2 * amps[..., 2:4, None]
    x = torch.sum(x_c1c2, dim=-2)

    return x


def _sample_params_mixed_symmetry(
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator],
):
    amplitudes = _sample_amplitudes(4, num_samples, device, generator)
    freqs = _sample_freqs(4, num_samples, device, generator)

    return freqs, amplitudes


def fm_hierarchical_symmetry(
    params: torch.Tensor, length: int, break_symmetry: bool = False
):
    """An FM synthesiser with algorithm ((M1+M2)->C1) + ((M3+M4)->C2)
    layout: [m1, m2, m3, m4, c1, c2]
    """
    freqs, amps = params.chunk(2, dim=-1)
    assert freqs.shape[-1] == 6
    assert amps.shape[-1] == 6
    n = torch.arange(length, device=freqs.device)
    # mod indices should go higher than amplitudes
    amps = amps.clone()
    amps[..., 0:4] *= 2
    freqs, amps = _scale_freqs_and_amps_fm(freqs, amps)

    if break_symmetry:
        shifts = torch.arange(4, device=freqs.device) * torch.pi / 4
        freqs[:, 0:4] = shifts + freqs[:, 0:4] / 4
        freqs[:, 4] = freqs[:, 4] / 2
        freqs[:, 5] = (freqs[:, 5] + torch.pi) / 2

    phi_m = freqs[..., :4, None] * n
    x_m = torch.sin(phi_m)
    x_m = x_m * amps[..., :4, None]
    x_m = x_m.view(*x_m.shape[:-2], 2, 2, -1).sum(-2)

    phi_c = freqs[..., 4:6, None] * n + x_m
    x_c = torch.sin(phi_c)
    x_c = x_c * amps[..., 4:6, None]
    x = torch.sum(x_c, dim=-2)

    return x


def _sample_params_hierarchical_symmetry(
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator],
):
    amplitudes = _sample_amplitudes(6, num_samples, device, generator)
    freqs = _sample_freqs(6, num_samples, device, generator)

    return freqs, amplitudes


_FM_ALGORITHMS = dict(
    conditional=(_sample_params_conditional_symmetry, fm_conditional_symmetry),
    mixed=(_sample_params_mixed_symmetry, fm_mixed_symmetry),
    hierarchical=(_sample_params_hierarchical_symmetry, fm_hierarchical_symmetry),
)


class FMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        algorithm: Literal["conditional", "mixed", "hierarchical"],
        signal_length: int,
        num_samples: int,
        break_symmetry: bool,
        seed: int,
    ):
        self.algorithm = algorithm
        self.signal_length = signal_length

        self.num_samples = num_samples

        self.break_symmetry = break_symmetry

        self.seed = seed
        self.generator = torch.Generator(device=torch.device("cpu"))

        self._init_dataset()

    def _init_dataset(self):
        self.generator.manual_seed(self.seed)
        freqs, amps = self._sample_parameters()
        freqs.share_memory_()
        amps.share_memory_()
        self.freqs = freqs
        self.amps = amps

    def _sample_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sampler, _ = _FM_ALGORITHMS[self.algorithm]
        freqs, amplitudes = sampler(
            self.num_samples, torch.device("cpu"), self.generator
        )

        return freqs, amplitudes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        freq = self.freqs[idx][None]
        amp = self.amps[idx][None]

        _, synth = _FM_ALGORITHMS[self.algorithm]
        synth = partial(
            synth, length=self.signal_length, break_symmetry=self.break_symmetry
        )

        signals = synth(freq, amp)
        params = torch.cat((freq, amp), dim=-1)

        return (signals, params, synth)


class FMDataModule(LightningDataModule):
    """The FM task is designed to probe conditional symmetry by constructing signals from simple
    frequency modulation synthesisers."""

    def __init__(
        self,
        algorithm: Literal["conditional", "mixed", "hierarchical"],
        signal_length: int = 1024,
        break_symmetry: bool = False,
        train_val_test_sizes: Tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: Tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 1024,
        num_workers: int = 0,
        ot: bool = False,
    ):
        super().__init__()

        # signal
        self.algorithm = algorithm
        self.signal_length = signal_length
        self.break_symmetry = break_symmetry

        # dataset
        self.train_size, self.val_size, self.test_size = train_val_test_sizes
        self.train_seed, self.val_seed, self.test_seed = train_val_test_seeds

        # dataloader
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.ot = ot

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            train_ds = FMDataset(
                self.algorithm,
                self.signal_length,
                self.train_size,
                self.break_symmetry,
                self.train_seed,
            )
            val_ds = FMDataset(
                self.algorithm,
                self.signal_length,
                self.val_size,
                self.break_symmetry,
                self.val_seed,
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
                collate_fn=ot_collate_fn if self.ot else regular_collate_fn,
                num_workers=self.num_workers,
            )
        else:
            test_ds = FMDataset(
                self.algorithm,
                self.signal_length,
                self.test_size,
                self.break_symmetry,
                self.test_seed,
            )
            self.test = torch.utils.data.DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=ot_collate_fn if self.ot else regular_collate_fn,
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
    dm = FMDataModule(k=4)
    dm.setup("fit")
    for x, y in dm.train:
        print(x.shape, y.shape)
        break
