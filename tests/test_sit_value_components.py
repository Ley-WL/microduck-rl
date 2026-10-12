import importlib.util
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/server'))
spec=importlib.util.spec_from_file_location('reward_audit',Path(__file__).resolve().parents[1]/'scripts/server/audit_sit_reward_gradients.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_current_baseline_future_remainder_reward_reconstruct_with_resets():
    rewards=torch.tensor([[[1.,2.],[2.,-1.]],[[3.,-1.],[1.,4.]],[[2.,4.],[3.,5.]]])
    values=torch.tensor([[.5,2.],[1.,3.],[2.,4.]])
    dones=torch.tensor([[0,0],[1,0],[0,1]])
    timeouts=torch.tensor([[0,0],[1,0],[0,1]])
    parts,normalized=audit.gae_parts(rewards,values,dones,timeouts,torch.tensor([3.,5.]),.99,.95)
    raw=parts.sum(-1);baseline=-values.double()
    components=torch.stack([baseline,parts[...,-1]-baseline,parts[...,:-1].sum(-1)],-1)
    normalized_components=(components-components.mean((0,1),keepdim=True))/(raw.std()+1e-8)
    torch.testing.assert_close(normalized_components.sum(-1),normalized.sum(-1),atol=1e-12,rtol=1e-12)

def test_state_only_baseline_mean_score_cancels_for_antithetic_actions():
    means=torch.tensor([.2,-.4],dtype=torch.double,requires_grad=True)
    actions=torch.stack([means.detach()+.1,means.detach()-.1])
    log=torch.distributions.Normal(means,.1).log_prob(actions)
    ratio=(log-log.detach()).exp()
    baseline=torch.tensor([3.,7.],dtype=torch.double)
    gradient=torch.autograd.grad(-(baseline*ratio).mean(),means)[0]
    torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-12,rtol=0)
