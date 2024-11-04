from typing import Optional, Type

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int],
        output_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize the MLP with given layer sizes and activation function."""
        super().__init__()

        layers: list[nn.Module] = []
        sizes = [input_dim] + list(hidden_dims)

        # Create the hidden layers with activation functions
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())

        # Add the final output layer
        layers.append(nn.Linear(sizes[-1], output_dim))

        # Wrap the layers in a Sequential container
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass through the network."""
        return self.net(x)


class Normalizer(torch.nn.Module):
    """
    Helper module for normalizing / unnormalzing torch tensors with a given mean and variance.
    """

    def __init__(self, mean: torch.tensor, variance: torch.tensor):
        """
        Initializes Normalizer object.

        Args:
            mean: torch.tensor with shape (n,); mean of normalization.
            variance: torch.tensor, shape (n,); variance of normalization.
        """
        super().__init__()

        # Check shapes -- need data to be 1D.
        assert len(mean.shape) == 1, ""
        assert len(variance.shape) == 1

        self.register_buffer("mean", mean)
        self.register_buffer("variance", variance)

    def normalize(self, x: torch.tensor):
        """
        Applies the normalization, i.e., computes (x - mean) / sqrt(variance).
        """
        assert x.shape[-1] == self.mean.shape[0]

        return (x - self.mean) / torch.sqrt(self.variance)

    def unnormalize(
        self,
        mean_normalized: torch.tensor,
        var_normalized: Optional[torch.tensor] = None,
    ):
        """
        Applies the unnormalization to the mean and variance, i.e., computes
        mean_normalized * sqrt(variance) + mean and var_normalized * variance.
        """
        assert mean_normalized.shape[-1] == self.mean.shape[0]
        if var_normalized is not None:
            assert var_normalized.shape[-1] == self.variance.shape[0]

        mean = mean_normalized * torch.sqrt(self.variance) + self.mean
        if var_normalized is not None:
            variance = var_normalized * self.variance
            return mean, variance
        else:
            return mean
