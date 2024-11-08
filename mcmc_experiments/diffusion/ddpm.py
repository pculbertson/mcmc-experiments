from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from mcmc_experiments.utils.models import MLP, Normalizer


@dataclass
class DDPMConfig:
    data_dim: int
    num_denoising_steps: int = 4000
    hidden_dims: tuple[int] = (32, 32)
    noise_schedule: Literal["cosine", "linear"] = "linear"


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        noise_schedule: str,
        num_denoising_steps: int,
        linear_beta_start: float = 1e-4,
        linear_beta_end: float = 2e-2,
        cosine_offset: float = 8e-3,
    ):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.num_denoising_steps = num_denoising_steps

        if noise_schedule == "linear":
            self.register_buffer(
                "betas",
                torch.linspace(linear_beta_start, linear_beta_end, num_denoising_steps),
            )
        elif noise_schedule == "cosine":
            timestep_frac = torch.arange(num_denoising_steps) / num_denoising_steps
            print(timestep_frac)
            cos_terms = torch.pow(
                torch.cos(
                    (torch.pi / 2)
                    * (timestep_frac + cosine_offset)
                    / (1 + cosine_offset)
                ),
                2,
            )
            alphas_cumprod = cos_terms / cos_terms[0]
            print(alphas_cumprod)
            self.register_buffer(
                "betas",
                torch.clip(
                    1
                    - alphas_cumprod
                    / nn.functional.pad(alphas_cumprod[:-1], (0, 1), "constant", 0.0),
                    0.0,
                    0.999,
                ),
            )
        else:
            raise ValueError("noise schedule must be cosine or linear.")
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, 0))

    def add_noise(
        self, x: torch.Tensor, raw_noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Adds noise to a sample for the given noise level."""
        selected_alpha_bars = self.alphas_cumprod[timesteps, None]
        return (
            torch.sqrt(selected_alpha_bars) * x
            + torch.sqrt(1 - selected_alpha_bars) * raw_noise
        )

    def denoise_step(
        self,
        x: torch.Tensor,
        pred_noise: torch.Tensor,
        raw_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        selected_alpha_bars = self.alphas_cumprod[timesteps].unsqueeze(-1)
        selected_alphas = self.alphas[timesteps].unsqueeze(-1)
        selected_betas = self.betas[timesteps].unsqueeze(-1)
        return (1 / torch.sqrt(selected_alphas)) * (
            x - (selected_betas / torch.sqrt(1 - selected_alpha_bars)) * pred_noise
        ) + torch.sqrt(selected_betas) * raw_noise


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_size = dim // 2
        self.freqs = torch.exp(
            -(torch.arange(half_size) / half_size) * torch.log(torch.tensor(1e4))
        )

    def forward(self, t: torch.Tensor):
        inputs = self.freqs.unsqueeze(0) * t.unsqueeze(-1)
        sines = torch.sin(inputs)
        cosines = torch.cos(inputs)

        return torch.stack([sines, cosines], dim=-1).flatten(-2, -1)


class DDPMBlock(nn.Module):
    def __init__(self, data_dim: int, hidden_dims: tuple[int]):
        super().__init__()
        self.positional_encoding = PositionalEncoding(data_dim)
        self.network = MLP(data_dim, hidden_dims, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        pos_embed = self.positional_encoding(t)
        x = self.network(x + pos_embed)

        return x


class DDPM(nn.Module):
    def __init__(
        self,
        config: DDPMConfig,
        data_shape: torch.Size,
        data_mean: Optional[torch.Tensor] = None,
        data_var: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_denoising_steps = config.num_denoising_steps
        self.ddpm_block = DDPMBlock(config.data_dim, config.hidden_dims)
        self.noise_scheduler = NoiseScheduler(
            config.noise_schedule, config.num_denoising_steps
        )
        self.data_shape = data_shape
        self.data_normalizer = Normalizer(data_mean, data_var)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x = self.data_normalizer.normalize(x)
        t = torch.randint(0, self.num_denoising_steps, (x.shape[0],))
        raw_noise = torch.randn_like(x)
        x_noised = self.noise_scheduler.add_noise(x, raw_noise, t)
        noise_pred = self.ddpm_block(x_noised, t)
        return (noise_pred - raw_noise).square().mean()

    def sample(self, num_samples: int) -> torch.Tensor:
        x = torch.randn(num_samples, *self.data_shape)

        for t in torch.arange(self.num_denoising_steps - 1, 0, -1):
            t = t.expand(
                num_samples,
            )
            pred_noise = self.ddpm_block(x, t)
            raw_noise = torch.randn_like(x)
            print(x.shape, pred_noise.shape)
            x = self.noise_scheduler.denoise_step(x, pred_noise, raw_noise, t)

        return self.data_normalizer.unnormalize(x)
