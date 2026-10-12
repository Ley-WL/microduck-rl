import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy
import torch
from tensordict import TensorDict

spec=importlib.util.spec_from_file_location('gradient_audit',Path(__file__).resolve().parents[1]/'scripts/server/audit_sit_gradient_contributions.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_clipping_matches_actual_torch_actor_clip():
    p=torch.nn.Parameter(torch.zeros(3));p.grad=torch.tensor([3.,4.,0.])
    expected=audit.clipped(p.grad.clone(),1.)
    torch.nn.utils.clip_grad_norm_([p],1.)
    torch.testing.assert_close(p.grad,expected)


def test_cosine_preserves_opposed_and_absent_directions():
    x=torch.tensor([3.,4.])
    assert abs(audit.cosine(x,-x)+1)<1e-6
    assert audit.cosine(x,torch.zeros_like(x)) is None


class Actor(torch.nn.Linear):
    def forward(self,obs):
        self.output_cache=super().forward(obs['actor'])
        return self.output_cache


def test_shadow_adam_uses_momentum_and_preserves_original():
    actor=Actor(2,2);critic=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=.01)
    for p in [*actor.parameters(),*critic.parameters()]:p.grad=torch.ones_like(p)
    optimizer.step()
    original=[p.detach().clone() for p in [*actor.parameters(),*critic.parameters()]]
    obs=TensorDict({'actor':torch.tensor([[1.,2.]])},batch_size=[1])
    actor(obs) # non-leaf output cache must copy without retaining its graph
    gradient=torch.zeros(sum(p.numel() for p in actor.parameters()))
    critic_gradient=torch.zeros(sum(p.numel() for p in critic.parameters()))
    report,displacement=audit.shadow_update(SimpleNamespace(actor=actor,critic=critic,optimizer=optimizer,learning_rate=.01),gradient,critic_gradient,obs)
    assert report['parameterDeltaNorm']>0 # inherited Adam momentum, despite zero new gradients
    for p,before in zip([*actor.parameters(),*critic.parameters()],original):torch.testing.assert_close(p,before,atol=0,rtol=0)
