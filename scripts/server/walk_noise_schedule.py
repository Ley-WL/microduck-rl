"""Fixed Gaussian noise within each PPO rollout/update; deterministic export unchanged."""
import torch


def sigma_at(iteration, mode):
    assert mode in ('fixed', 'anneal')
    if mode == 'fixed':
        return .05
    # 200 x 24 x .02 = 96 seconds: warmup and final plateau each exceed a cycle.
    fraction = min(1., max(0., (iteration - 65) / 64.))
    return .05 + (.005 - .05) * fraction


class RolloutNoise:
    def __init__(self, algorithm, mode, steps=24):
        from rsl_rl.modules.distribution import GaussianDistribution
        distribution = algorithm.actor.distribution
        assert type(distribution) is GaussianDistribution
        assert distribution.std_type == 'scalar'
        self.parameter = distribution.std_param
        self.parameter.requires_grad_(False)
        self.mode, self.steps = mode, steps
        self.iteration = self.calls = 0
        self.audit = []
        self.original_act, self.original_update = algorithm.act, algorithm.update
        algorithm.act, algorithm.update = self.act, self.update

    def act(self, observations):
        if self.calls == 0:
            with torch.no_grad():
                self.parameter.fill_(sigma_at(self.iteration, self.mode))
            self.expected = self.parameter.detach().clone()
        assert torch.equal(self.parameter, self.expected)
        self.calls += 1
        assert self.calls <= self.steps
        return self.original_act(observations)

    def update(self):
        assert self.calls == self.steps
        assert torch.equal(self.parameter, self.expected)
        result = self.original_update()
        assert torch.equal(self.parameter, self.expected), 'Noise changed inside PPO update'
        self.audit.append(dict(iteration=self.iteration, sigma=float(self.parameter[0]),
                               rolloutSteps=self.calls, constantThroughUpdate=True))
        self.iteration += 1
        self.calls = 0
        return result
