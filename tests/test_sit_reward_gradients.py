import importlib.util
from pathlib import Path
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/server"))
spec=importlib.util.spec_from_file_location('reward_audit',Path(__file__).resolve().parents[1]/'scripts/server/audit_sit_reward_gradients.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_rewards_plus_value_and_timeouts_reconstruct_actual_gae():
    r=torch.tensor([[[1.,2.]],[[3.,-1.]],[[2.,4.]]]);v=torch.tensor([[.5],[1.],[2.]])
    dones=torch.tensor([[0],[1],[0]]);timeouts=torch.tensor([[0],[1],[0]]);last=torch.tensor([3.])
    parts,normalized=audit.gae_parts(r,v,dones,timeouts,last,.99,.95)
    raw=torch.zeros_like(v);future=0.
    for t in (2,1,0):
        next_value=last if t==2 else v[t+1]
        delta=r[t].sum(-1)+.99*v[t]*timeouts[t]+.99*next_value*(1-dones[t])-v[t]
        future=delta+.99*.95*(1-dones[t])*future;raw[t]=future
    torch.testing.assert_close(parts.sum(-1).float(),raw)
    torch.testing.assert_close(normalized.sum(-1).float(),(raw-raw.mean())/(raw.std()+1e-8))

def test_single_common_normalization_preserves_canceling_components():
    r=torch.tensor([[[100.,-99.]],[[200.,-198.]],[[300.,-296.]]]);v=torch.zeros(3,1)
    parts,n=audit.gae_parts(r,v,torch.zeros(3,1),torch.zeros(3,1),torch.zeros(1),.9,.9)
    total=parts.sum(-1);expected=(total-total.mean())/(total.std()+1e-8)
    torch.testing.assert_close(n.sum(-1),expected)
