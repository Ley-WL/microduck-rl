"""PPO actor retention + measured adjacent-input regularization.

Add auxiliary gradients to the same step, then clip the combined gradient.
The anchor uses its own frozen normalizer. Never zero posture command slots.
"""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from .symmetry import PpoWithSymmetryCfg


@dataclass
class SitTransferPpoCfg(PpoWithSymmetryCfg):
    class_name:str='mjlab_microduck.tasks.sit_transfer_ppo.SitTransferPpo'
    retention_checkpoint:str='seed.pt'
    transfer_data:str='transfer-data.npz'
    retention_coef:float=10.
    temporal_coef:float=.3
    onpolicy_retention_coef:float=2.
    exploration_std_range:tuple[float,float]=(.05,.15)


def clamp_exploration_std(distribution,bounds):
    low,high=bounds
    assert 0<low<=high
    with torch.no_grad():distribution.std_param.clamp_(low,high)


class SitTransferPpo(PPO):
    def __init__(self,actor,critic,storage,*args,retention_checkpoint,transfer_data,retention_coef,temporal_coef,onpolicy_retention_coef=2.,exploration_std_range=(.05,.15),**kwargs):
        super().__init__(actor,critic,storage,*args,**kwargs)
        state=torch.load(retention_checkpoint,map_location=self.device,weights_only=False)
        self.anchor=copy.deepcopy(actor).to(self.device)
        self.anchor.load_state_dict(state['actor_state_dict'],strict=True)
        self.anchor.eval()
        for p in self.anchor.parameters():p.requires_grad_(False)
        data=np.load(transfer_data)
        self.demonstrations=torch.tensor(data['simulation'],device=self.device)
        self.real=torch.tensor(data['real'],device=self.device)
        self.pairs=torch.tensor(data['pair_indexes'],device=self.device,dtype=torch.long)
        assert self.demonstrations.shape[1]==61 and self.real.shape[1]==61
        assert list(self.actor.obs_groups)==['actor']
        self.retention_coef=retention_coef;self.temporal_coef=temporal_coef
        self.onpolicy_retention_coef=onpolicy_retention_coef
        assert 0<exploration_std_range[0]<=exploration_std_range[1]
        self.exploration_std_range=exploration_std_range
        self._stats=[]
        self._aux_hook=self.optimizer.register_step_pre_hook(self._add_auxiliary)
        print('[sit-transfer] frozen seed anchor + temporal physical-input pairs; actor normalization frozen')

    def _td(self,x):return TensorDict({'actor':x},batch_size=[len(x)])

    def train_mode(self):
        super().train_mode()
        self.actor.obs_normalizer.eval()

    def act(self,obs):
        # Loading the warm-start checkpoint restores its old std after __init__.
        # Clamp before the first rollout as well as after each optimizer update.
        clamp_exploration_std(self.actor.distribution,self.exploration_std_range)
        return super().act(obs)

    def _add_auxiliary(self,optimizer,args,kwargs):
        ids=torch.randint(len(self.demonstrations),(512,),device=self.device)
        x=self.demonstrations[ids]
        onpolicy=self.storage.observations['actor'].flatten(0,1)
        current=onpolicy[torch.randint(len(onpolicy),(512,),device=self.device)]
        with torch.no_grad():
            expected=self.anchor(self._td(x));expected_current=self.anchor(self._td(current))
        retain=(self.actor(self._td(x))-expected).square().mean()
        current_retain=(self.actor(self._td(current))-expected_current).square().mean()
        prediction=self.actor(self._td(self.real))
        temporal=(prediction[self.pairs]-prediction[self.pairs-1]).square().mean()
        loss=self.retention_coef*retain+self.onpolicy_retention_coef*current_retain+self.temporal_coef*temporal
        loss.backward()
        # PPO has already clipped its own gradients; clip again after adding
        # retention so the actual combined step respects max_grad_norm.
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
        self._stats.append((float(retain.detach()),float(current_retain.detach()),float(temporal.detach())))

    def update(self):
        self._stats=[]
        result=super().update()
        clamp_exploration_std(self.actor.distribution,self.exploration_std_range)
        if self._stats:
            values=np.mean(self._stats,axis=0)
            result.update(sit_retention=float(values[0]),sit_onpolicy_retention=float(values[1]),sit_real_temporal=float(values[2]))
        return result
