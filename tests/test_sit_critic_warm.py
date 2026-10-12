from types import SimpleNamespace
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.tasks.sit_critic_warm import SitCriticWarm


def test_critic_only_calibration_improves_values_without_touching_actor():
    torch.manual_seed(4)
    alg=SitCriticWarm.__new__(SitCriticWarm)
    alg.actor=torch.nn.Linear(2,2)
    alg.critic=torch.nn.Linear(2,1)
    alg.critic_warm_optimizer=torch.optim.Adam(alg.critic.parameters(),lr=.03)
    alg.critic_warm_updates=0;alg.critic_warmup_iterations=20
    alg.num_mini_batches=1;alg.num_learning_epochs=1;alg.max_grad_norm=1.
    x=torch.randn(64,2);y=x[:,0:1]*2-x[:,1:2]
    batch=SimpleNamespace(observations=x,returns=y)
    cleared=[]
    alg.storage=SimpleNamespace(mini_batch_generator=lambda *_:[batch],clear=lambda:cleared.append(True))
    actor={k:v.clone() for k,v in alg.actor.state_dict().items()}
    initial=float((alg.critic(x)-y).square().mean().detach())
    for _ in range(20):alg._update_critic_only()
    assert float((alg.critic(x)-y).square().mean().detach())<initial*.5
    assert len(cleared)==20 and alg.critic_warm_updates==20
    for k,v in actor.items():torch.testing.assert_close(v,alg.actor.state_dict()[k],atol=0,rtol=0)


def test_critic_warm_keeps_rewards_motion_and_matched_ppo_budget():
    old='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910'
    new='Mjlab-SitStandProcessCriticWarm-Flat-MicroDuck-HD1910'
    a,b=load_env_cfg(old),load_env_cfg(new)
    for field in ('actions','observations','commands','events','rewards','curriculum','terminations'):
        assert getattr(a,field)==getattr(b,field)
    a,b=load_rl_cfg(old),load_rl_cfg(new)
    assert b.max_iterations-b.algorithm.critic_warmup_iterations==a.max_iterations==100
    for k,v in vars(a.algorithm).items():
        if k!='class_name':assert getattr(b.algorithm,k)==v
