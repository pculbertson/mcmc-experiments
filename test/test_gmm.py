import torch
from torch.distributions import MultivariateNormal
from mcmc_experiments.distributions import GaussianMixtureModel


def test_single_component_gmm():
    """Test that a single-component GMM behaves like a standard Gaussian."""
    mean = torch.tensor([0.0, 0.0])
    covariance = torch.eye(2)
    weights = torch.tensor([1.0])  # Single component, full weight

    # Create the GMM and a corresponding MultivariateNormal distribution
    gmm = GaussianMixtureModel(mean.unsqueeze(0), covariance.unsqueeze(0), weights)
    mvn = MultivariateNormal(mean, covariance)

    # Generate some test data points
    data = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    # Check that the GMM likelihood matches that of the single Gaussian component
    assert torch.allclose(gmm.likelihood(data), mvn.log_prob(data).exp(), atol=1e-6)

    # Check that the GMM score matches the gradient of the single Gaussian
    data.requires_grad_(True)
    mvn_log_prob = mvn.log_prob(data).sum()
    mvn_log_prob.backward()
    assert torch.allclose(
        gmm.score(data), data.grad, atol=1e-6
    ), f"Scores did not match, got {gmm.score(data)} vs. {data.grad}"


def test_multiple_components_likelihood_shape():
    """Test that the GMM likelihood has the correct shape for multiple components."""
    means = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    covariances = torch.stack([torch.eye(2), torch.eye(2)])
    weights = torch.tensor([0.5, 0.5])

    gmm = GaussianMixtureModel(means, covariances, weights)

    # Generate some test data points
    data = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])

    # Check that the likelihood output has the expected shape
    likelihoods = gmm.likelihood(data)
    assert likelihoods.shape == (3,)
    assert torch.all(likelihoods > 0)  # Likelihoods should be positive


def test_gmm_sample_shape():
    """Test that the GMM generates samples with the correct shape."""
    means = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    covariances = torch.stack([torch.eye(2), torch.eye(2)])
    weights = torch.tensor([0.5, 0.5])

    gmm = GaussianMixtureModel(means, covariances, weights)

    # Draw 1000 samples and check the shape
    samples = gmm.sample(1000)
    assert samples.shape == (1000, 2)


def test_weighted_sampling():
    """Test that different components are sampled according to their weights."""
    means = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    covariances = torch.stack([torch.eye(2), torch.eye(2)])
    weights = torch.tensor([0.9, 0.1])  # Heavily favor the first component

    gmm = GaussianMixtureModel(means, covariances, weights)

    # Draw 10000 samples and check the proportion of samples near each mean
    samples = gmm.sample(10000)

    # Compute the number of samples close to each mean (within 3 standard deviations)
    near_first_mean = (samples - means[0]).norm(dim=-1) < 3
    near_second_mean = (samples - means[1]).norm(dim=-1) < 3

    # Assert that more samples are closer to the first mean than the second
    assert near_first_mean.sum() > 8500  # At least 85% from first component
    assert near_second_mean.sum() < 1500  # No more than 15% from second component
