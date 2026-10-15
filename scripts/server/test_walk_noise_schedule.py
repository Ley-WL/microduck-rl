"""Exercise real Gaussian log probabilities across rollout boundaries and updates."""
from types import SimpleNamespace
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from walk_noise_schedule import RolloutNoise, sigma_at


for mode in ('fixed','anneal','low'):
    dist=GaussianDistribution(14,init_std=.05)
    saved=[]
    def act(obs):
        dist.update(obs)
        action=dist.sample()
        saved.append((obs,action,dist.log_prob(action).clone()))
        return action
    def update():
        for obs,action,old in saved:
            dist.update(obs)
            assert torch.equal(dist.log_prob(action),old)
        saved.clear()
        return {}
    alg=SimpleNamespace(actor=SimpleNamespace(distribution=dist),act=act,update=update)
    noise=RolloutNoise(alg,mode)
    for iteration in range(200):
        for step in range(24):alg.act(torch.zeros(2,14))
        alg.update()
        assert abs(float(dist.std_param[0])-sigma_at(iteration,mode))<1e-8
    assert not dist.std_param.requires_grad
    assert noise.iteration==200 and noise.calls==0
assert sigma_at(0,'anneal')==.05 and abs(sigma_at(129,'anneal')-.005)<1e-9
assert all(sigma_at(i,'low')==.005 for i in (0,5,65,129,199,400))
print('200 rollout/update boundaries: exact log-probability parity, fixed/annealed/low schedules passed')

# Resumed rollouts keep the final exploration level, including across updates.
alg=SimpleNamespace(actor=SimpleNamespace(distribution=dist),act=act,update=update)
noise=RolloutNoise(alg,'anneal');noise.iteration=200
for iteration in range(200,205):
    for step in range(24):alg.act(torch.zeros(2,14))
    alg.update()
    assert abs(float(dist.std_param[0])-.005)<1e-8
assert [r['iteration'] for r in noise.audit]==list(range(200,205))
print('Resume offset 200 preserves .005 noise and exact log probabilities')
