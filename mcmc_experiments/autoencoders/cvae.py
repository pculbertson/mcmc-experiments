from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence

from mcmc_experiments.utils.models import MLP, Normalizer


@dataclass
class CVAEConfig:
    input_dim: int
    conditioning_dim: int
    latent_dim: int
    encoder_layers: Tuple[int]
    decoder_layers: Tuple[int]
    prior_layers: Tuple[int]
    loss_type: Literal["elbo", "iwae"] = "elbo"
    elbo_kl_weight: float = 1.0
    num_iwae_samples: int = 20


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with conditioning
    and output data normalization.
    """

    def __init__(
        self,
        config: CVAEConfig,
        cond_mean: Optional[Tensor] = None,
        cond_var: Optional[Tensor] = None,
        output_mean: Optional[Tensor] = None,
        output_var: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.input_dim = config.input_dim
        self.conditioning_dim = config.conditioning_dim
        self.latent_dim = config.latent_dim

        self.loss_type = config.loss_type
        self.elbo_kl_weight = config.elbo_kl_weight
        self.num_iwae_samples = config.num_iwae_samples

        # Initialize normalization layers with default values.
        self.cond_normalizer = self._create_normalizer(
            cond_mean, cond_var, self.conditioning_dim
        )
        self.output_normalizer = self._create_normalizer(
            output_mean, output_var, self.input_dim
        )

        # Define encoder, decoder, and prior networks.
        self.encoder = MLP(
            input_dim=self.input_dim + self.conditioning_dim,
            output_dim=2 * self.latent_dim,
            hidden_dims=config.encoder_layers,
        )
        self.decoder = MLP(
            input_dim=self.latent_dim + self.conditioning_dim,
            output_dim=2 * self.input_dim,
            hidden_dims=config.decoder_layers,
        )
        self.prior_network = MLP(
            input_dim=self.conditioning_dim,
            output_dim=2 * self.latent_dim,
            hidden_dims=config.prior_layers,
        )

    def _create_normalizer(
        self, mean: Optional[Tensor], var: Optional[Tensor], dim: int
    ) -> Normalizer:
        """Create a normalizer with default mean/variance if not provided."""
        mean = mean if mean is not None else torch.zeros(dim)
        var = var if var is not None else torch.ones(dim)
        return Normalizer(mean, var)

    def encode(self, input: Tensor, conditioning: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes the input and returns latent mean and variance."""
        z_params = self.encoder(torch.cat([input, conditioning], dim=-1))
        z_mean, z_logvar = torch.chunk(z_params, 2, dim=-1)
        return z_mean, torch.exp(z_logvar)

    def decode(
        self, latent_vars: Tensor, conditioning: Tensor, unnormalize: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Decodes latent variables into output mean and variance."""
        output_params = self.decoder(torch.cat([latent_vars, conditioning], dim=-1))
        mean_norm, logvar_norm = torch.chunk(output_params, 2, dim=-1)
        variance_norm = torch.exp(logvar_norm)

        if unnormalize:
            return self.output_normalizer.unnormalize(mean_norm, variance_norm)
        return mean_norm, variance_norm

    def prior(self, conditioning: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns prior mean and variance for the given conditioning."""
        prior_params = self.prior_network(conditioning)
        prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
        return prior_mean, torch.exp(prior_logvar)

    def elbo_loss(self, data: Tensor, conditioning: Tensor) -> Tensor:
        """
        Computes the Evidence Lower Bound (ELBO) loss.

        Args:
            model: CVAE; the CVAE model.
            data: Tensor of shape (batch_size, output_dim); the target data.
            conditioning: Tensor of shape (batch_size, cond_dim); the conditioning data.

        Returns:
            Tensor; the mean ELBO loss.
        """
        # Normalize data and conditioning inputs.
        data_norm = self.output_normalizer.normalize(data)
        cond_norm = self.cond_normalizer.normalize(conditioning)

        # Encode data to get latent mean and variance.
        z_mean, z_var = self.encode(data_norm, cond_norm)

        # Reparameterize to get latent sample.
        z = reparameterize(z_mean, z_var)

        # Decode latent sample to get output mean and variance.
        output_mean, output_var = self.decode(z, cond_norm, unnormalize=False)

        # Compute reconstruction log probability (reconstruction loss).
        recon_loss = compute_log_prob(output_mean, output_var, data).mean()

        # Compute prior distribution and KL divergence.
        prior_mean, prior_var = self.prior(cond_norm)
        latent_dist = Normal(z_mean, torch.sqrt(z_var))
        prior_dist = Normal(prior_mean, torch.sqrt(prior_var))
        kl_div = (
            kl_divergence(latent_dist, prior_dist).sum(dim=-1).mean()
        )  # IMPORTANT: sum for KL div of multivariate Gaussian.

        # ELBO = Reconstruction Loss - KL Divergence
        elbo = recon_loss - kl_div

        return -elbo  # Return negative ELBO as the loss


def reparameterize(mean: Tensor, var: Tensor) -> Tensor:
    """Reparameterize a normal distribution to sample with rsample()."""
    std = torch.sqrt(var)
    eps = torch.randn_like(std)  # Sample from standard normal
    return mean + eps * std  # Reparameterized sample


def compute_log_prob(mean: Tensor, var: Tensor, data: Tensor) -> Tensor:
    """Compute log probability of data under Normal(mean, var)."""
    dist = Normal(mean, torch.sqrt(var))
    return dist.log_prob(data).sum(dim=-1)  # Sum over data dimensions
