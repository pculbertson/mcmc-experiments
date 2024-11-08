from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence

from mcmc_experiments.utils.models import MLP, Normalizer


@dataclass
class VAEConfig:
    input_dim: int
    latent_dim: int
    encoder_layers: Tuple[int]
    decoder_layers: Tuple[int]


class VAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with conditioning
    and output data normalization.
    """

    def __init__(
        self,
        config: VAEConfig,
        output_mean: Optional[Tensor] = None,
        output_var: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim

        # Initialize normalization layers with default values.
        self.output_normalizer = self._create_normalizer(
            output_mean, output_var, self.input_dim
        )

        # Define encoder, decoder, and prior networks.
        self.encoder = MLP(
            input_dim=self.input_dim,
            output_dim=2 * self.latent_dim,
            hidden_dims=config.encoder_layers,
        )
        self.decoder = MLP(
            input_dim=self.latent_dim,
            output_dim=2 * self.input_dim,
            hidden_dims=config.decoder_layers,
        )

    def _create_normalizer(
        self, mean: Optional[Tensor], var: Optional[Tensor], dim: int
    ) -> Normalizer:
        """Create a normalizer with default mean/variance if not provided."""
        mean = mean if mean is not None else torch.zeros(dim)
        var = var if var is not None else torch.ones(dim)
        return Normalizer(mean, var)

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes the input and returns latent mean and variance."""
        z_params = self.encoder(input)
        z_mean, z_logvar = torch.chunk(z_params, 2, dim=-1)
        return z_mean, torch.exp(z_logvar)

    def decode(
        self, latent_vars: Tensor, unnormalize: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Decodes latent variables into output mean and variance."""
        output_params = self.decoder(latent_vars)
        mean_norm, logvar_norm = torch.chunk(output_params, 2, dim=-1)
        variance_norm = torch.exp(logvar_norm)

        if unnormalize:
            return self.output_normalizer.unnormalize(mean_norm, variance_norm)
        return mean_norm, variance_norm

    def loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the Evidence Lower Bound (ELBO) loss.

        Args:
            model: CVAE; the CVAE self.
            data: Tensor of shape (batch_size, output_dim); the target data.
            conditioning: Tensor of shape (batch_size, cond_dim); the conditioning data.

        Returns:
            Tensor; the mean ELBO loss.
        """

        # Normalize data and conditioning inputs.
        data_norm = self.output_normalizer.normalize(data)

        # Encode data to get latent mean and variance.
        z_mean, z_var = self.encode(data_norm)

        # Reparameterize to get latent sample.
        z = reparameterize(z_mean, z_var)

        # Decode latent sample to get output mean and variance.
        output_mean, output_var = self.decode(z, unnormalize=False)

        # Compute reconstruction log probability (reconstruction loss).
        recon_loss = compute_log_prob(output_mean, output_var, data_norm).mean()

        # Compute prior distribution and KL divergence.
        latent_dist = Normal(z_mean, torch.sqrt(z_var))
        prior_dist = Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        kl_div = kl_divergence(latent_dist, prior_dist).mean()

        # ELBO = Reconstruction Loss - KL Divergence
        elbo = recon_loss - kl_div

        return -elbo  # Return negative ELBO as the loss

    def sample(self, num_samples: int) -> torch.Tensor:
        """Draws `num_samples` samples from the VAE."""
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z, unnormalize=True)


def reparameterize(mean: Tensor, var: Tensor) -> Tensor:
    """Reparameterize a normal distribution to sample with rsample()."""
    std = torch.sqrt(var)
    eps = torch.randn_like(std)  # Sample from standard normal
    return mean + eps * std  # Reparameterized sample


def compute_log_prob(mean: Tensor, var: Tensor, data: Tensor) -> Tensor:
    """Compute log probability of data under Normal(mean, var)."""
    dist = Normal(mean, torch.sqrt(var))
    return dist.log_prob(data).sum(dim=-1)  # Sum over data dimensions
