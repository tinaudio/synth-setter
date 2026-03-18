import math
from dataclasses import dataclass

import rootutils
import torch
import torch.nn as nn
from nflows.flows import SimpleRealNVP
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import param_specs  # noqa
from src.data.vst.param_spec import (
    CategoricalParameter,
    DiscreteLiteralParameter,
    Parameter,
)


class CustomRealNVP(CompositeTransform):
    """Taken from https://github.com/gwendal-lv/preset-gen-vae/blob/main/model/flows.py
    Le Vaillant et al"""

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=nn.functional.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        use_dropout = (
            True  # Quick and dirty: 'global' variable, as seen by the create_resnet function
        )

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability if use_dropout else 0.0,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for layer in range(num_layers):
            use_dropout = layer < (num_layers - 2)  # No dropout on the 2 last layers
            transform = coupling_constructor(mask=mask, transform_net_create_fn=create_resnet)
            layers.append(transform)
            mask *= -1  # Checkerboard masking inverse
            if batch_norm_between_layers and layer < (
                num_layers - 2
            ):  # No batch norm on the last 2 layers
                layers.append(BatchNorm(features=features))

        super().__init__(layers)


class EncoderBlock(nn.Module):
    """Like https://github.com/gwendal-lv/preset-gen-vae/blob/main/model/encoder.py
    but with added residual connections because that's what we do now."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: tuple[int],
        padding: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.inner_layers = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                (1, 1),
                padding,
                groups=out_channels,
            ),
        )
        self.final_act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inner_layers(x) + x
        x = self.final_act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, spec_dim: tuple[int] = (128, 401)):
        super().__init__()

        self.cnn = nn.Sequential(
            EncoderBlock(2, 16, (3, 5), (2, 2), (1, 2)),
            EncoderBlock(16, 32, (3, 5), (2, 2), (1, 2)),
            EncoderBlock(32, 64, (3, 5), (2, 2), (1, 2)),
            EncoderBlock(64, 64, (3, 5), (2, 2), (1, 2)),
            EncoderBlock(64, 128, (3, 5), (2, 2), (1, 2)),
            EncoderBlock(128, 256, (3, 5), (1, 4), (1, 2)),
            EncoderBlock(256, 512, (5, 5), (2, 2), 2),
            nn.Conv2d(512, 512, (1, 1), (1, 1), 0),
            nn.LeakyReLU(0.1),
        )

        dummy_spec = torch.randn(1, 2, *spec_dim)
        dummy_spec = self.cnn(dummy_spec)
        dummy_spec = dummy_spec.view(dummy_spec.shape[0], -1)
        num_features = dummy_spec.shape[1]

        self.out = nn.Sequential(
            nn.Linear(num_features, latent_dim * 2), nn.BatchNorm1d(latent_dim * 2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x.squeeze(1)


class DecoderBlock(nn.Module):
    """Like https://github.com/gwendal-lv/preset-gen-vae/blob/main/model/decoder.py
    but with added residual connections because that's what we do now."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: tuple[int],
        padding: int,
        output_padding: tuple[int] | None = None,
    ):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding=0 if output_padding is None else output_padding,
        )
        self.inner_layers = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), 0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                (1, 1),
                padding,
                groups=out_channels,
            ),
        )
        self.final_act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inner_layers(x) + x
        x = self.final_act(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, num_channels: int, spec_dim: tuple[int] = (128, 401)):
        super().__init__()

        self.in_proj = nn.Linear(latent_dim, 2048)
        self.cnn = nn.Sequential(
            DecoderBlock(512, 512, (5, 5), (2, 2), 2, (1, 1)),
            DecoderBlock(512, 256, (3, 5), (1, 4), (1, 2), output_padding=(0, 0)),
            DecoderBlock(256, 128, (3, 5), (2, 2), (1, 2), output_padding=(1, 1)),
            DecoderBlock(128, 64, (3, 5), (2, 2), (1, 2), output_padding=(1, 0)),
            DecoderBlock(64, 32, (3, 5), (2, 2), (1, 2), output_padding=(1, 0)),
            DecoderBlock(32, 16, (3, 5), (2, 2), (1, 2), output_padding=(1, 0)),
            DecoderBlock(16, 2, (3, 5), (2, 2), (1, 2), output_padding=(1, 0)),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = x.reshape(-1, 512, 2, 2)
        x = self.cnn(x)
        return x


@dataclass
class VAEOutput:
    y_hat: torch.Tensor

    x_hat: torch.Tensor

    z_0: torch.Tensor
    z_k: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    log_det_jacobian: torch.Tensor

    @property
    def shapes(self):
        return {
            "y_hat": self.y_hat.shape,
            "z_0": self.z_0.shape,
            "z_k": self.z_k.shape,
            "mu": self.mu.shape,
            "log_var": self.log_var.shape,
            "x_hat": self.x_hat.shape,
            "log_det_jacobian": self.log_det_jacobian.shape,
        }


class FlowVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 92,
        latent_flow_hidden_dim: int = 512,
        latent_flow_num_layers: int = 6,
        latent_flow_num_blocks: int = 2,
        latent_flow_batch_norm_within_layers: bool = True,
        latent_flow_batch_norm_between_layers: bool = False,
        regression_flow_hidden_dim: int = 512,
        regression_flow_num_layers: int = 16,
        regression_flow_num_blocks: int = 2,
        regression_flow_batch_norm_within_layers: bool = True,
        regression_flow_batch_norm_between_layers: bool = False,
        regression_flow_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.latent_flow = SimpleRealNVP(
            features=latent_dim,
            hidden_features=latent_flow_hidden_dim,
            num_layers=latent_flow_num_layers,
            num_blocks_per_layer=latent_flow_num_blocks,
            batch_norm_within_layers=latent_flow_batch_norm_within_layers,
            batch_norm_between_layers=latent_flow_batch_norm_between_layers,
        )

        self.regression_flow = CustomRealNVP(
            features=latent_dim,
            hidden_features=regression_flow_hidden_dim,
            num_layers=regression_flow_num_layers,
            num_blocks_per_layer=regression_flow_num_blocks,
            batch_norm_within_layers=regression_flow_batch_norm_within_layers,
            batch_norm_between_layers=regression_flow_batch_norm_between_layers,
            dropout_probability=regression_flow_dropout,
        )

    def forward(self, y: torch.Tensor) -> VAEOutput:
        latent = self.encoder(y)
        mu, log_var = latent.chunk(2, dim=-1)
        eps = torch.randn_like(mu)

        if self.training:
            z_0 = mu + eps * torch.exp(0.5 * log_var)
        else:
            z_0 = mu

        z_k, log_det_jacobian = self.latent_flow._transform(z_0)
        y_hat = self.decoder(z_k)

        x_hat, _ = self.regression_flow(z_k)

        return VAEOutput(y_hat, x_hat, z_0, z_k, mu, log_var, log_det_jacobian)


def reconstruction_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(y_hat, y)


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        x.shape[1] * math.log(2 * math.pi)
        + torch.sum(log_var + (x - mu).pow(2) / log_var.exp(), dim=1)
    )


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (x.shape[1] * math.log(2 * math.pi) + torch.sum(x.pow(2), dim=1))


def latent_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    z_0: torch.Tensor,
    z_k: torch.Tensor,
    log_det_jacobian: torch.Tensor,
) -> torch.Tensor:
    log_q_z0 = gaussian_log_prob(z_0, mu, log_var)
    log_p_zk = standard_normal_log_prob(z_k)
    # 0.2 factor is empirical from le vaillant paper
    return -torch.mean(log_p_zk - log_q_z0 + log_det_jacobian) * 0.2


def compute_individual_parameter_loss(
    x_hat: torch.Tensor, x: torch.Tensor, parameter: Parameter
) -> torch.Tensor:
    if (
        isinstance(parameter, DiscreteLiteralParameter)
        or isinstance(parameter, CategoricalParameter)
    ) and parameter.encoding == "onehot":
        labels = x.argmax(dim=1)
        loss = nn.functional.cross_entropy(x_hat, labels, reduction="mean")
    else:
        loss = nn.functional.mse_loss(x_hat, x)

    return loss


def param_loss(x_hat: torch.Tensor, x: torch.Tensor, param_spec: str) -> torch.Tensor:
    param_spec = param_specs[param_spec]

    synth_params = [(p, len(p)) for p in param_spec.synth_params]
    note_params = [(p, len(p)) for p in param_spec.note_params]

    pointer = 0

    loss = 0.0

    for param, length in synth_params:
        x_param = x[:, pointer : pointer + length]
        x_hat_param = x_hat[:, pointer : pointer + length]

        this_loss = compute_individual_parameter_loss(x_hat_param, x_param, param)

        loss += this_loss

        pointer += length

    for param, length in note_params:
        x_param = x[:, pointer : pointer + length]
        x_hat_param = x_hat[:, pointer : pointer + length]

        this_loss = compute_individual_parameter_loss(x_hat_param, x_param, param)

        loss += this_loss

        pointer += length

    assert pointer == x.shape[1]

    return loss / (len(synth_params) + len(note_params))


def compute_flowvae_loss(
    vae_output: VAEOutput,
    y: torch.Tensor,
    x: torch.Tensor,
    param_spec: str = "surge_xt",
) -> dict[str, torch.Tensor]:
    return dict(
        reconstruction_loss=reconstruction_loss(vae_output.y_hat, y),
        latent_loss=latent_loss(
            vae_output.mu,
            vae_output.log_var,
            vae_output.z_0,
            vae_output.z_k,
            vae_output.log_det_jacobian,
        ),
        param_loss=param_loss(vae_output.x_hat, x, param_spec),
    )


if __name__ == "__main__":
    params = 92
    enc = Encoder(params, (128, 401))
    dec = Decoder(params, 2, (128, 401))
    vae = FlowVAE(enc, dec, params)
    fake_spec = torch.randn(9, 2, 128, 401)
    fake_params = torch.rand(9, params)
    out = vae(fake_spec)
    losses = compute_flowvae_loss(out, fake_spec, fake_params)
    print(losses)
    print(sum(p.numel() for p in vae.parameters()))
