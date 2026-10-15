"""Equal-covariance Gaussian KL to the frozen actor, with a fixed noise scale."""
import torch


def gaussian_mean_kl(prediction, target, sigma):
    assert prediction.ndim == 2 and prediction.shape == target.shape
    assert sigma.shape == prediction.shape[-1:]
    assert torch.isfinite(sigma).all() and (sigma > 0).all()
    # Sum over actions, average over observations: matches distribution KL.
    # Sigma is the fixed exploration scale, not a parameter the actor can enlarge.
    return .5 * ((prediction-target.detach())/sigma.detach()).square().sum(-1).mean()
