"""Bounded critic-only calibration before matched actor PPO updates."""
from dataclasses import dataclass
import torch
from .sit_transfer_ppo import SitTransferPpo,SitTransferPpoCfg


@dataclass
class SitCriticWarmCfg(SitTransferPpoCfg):
    class_name:str='mjlab_microduck.tasks.sit_critic_warm.SitCriticWarm'
    critic_warmup_iterations:int=20
    critic_warmup_lr:float=1e-3


class SitCriticWarm(SitTransferPpo):
    def __init__(self,*args,critic_warmup_iterations=20,critic_warmup_lr=1e-3,**kwargs):
        super().__init__(*args,**kwargs)
        assert critic_warmup_iterations>=0 and critic_warmup_lr>0
        self.critic_warmup_iterations=critic_warmup_iterations
        self.critic_warm_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_warmup_lr)
        self.critic_warm_updates=0

    def _update_critic_only(self):
        # No main optimizer step: its actor/auxiliary hook cannot run here.
        before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        losses=[]
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs):
            loss=(self.critic(batch.observations)-batch.returns.detach()).square().mean()
            assert bool(torch.isfinite(loss))
            self.critic_warm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
            self.critic_warm_optimizer.step()
            losses.append(float(loss.detach()))
        for k,v in self.actor.state_dict().items():
            torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        self.storage.clear()
        self.critic_warm_updates+=1
        value=sum(losses)/len(losses)
        print(f'[critic-warm] {self.critic_warm_updates}/{self.critic_warmup_iterations} value_mse={value:.6g} actor_exactly_preserved=True',flush=True)
        return {'value':value,'surrogate':0.,'entropy':0.,'critic_warm_actor_preserved':1.}

    def update(self):
        if self.critic_warm_updates<self.critic_warmup_iterations:return self._update_critic_only()
        return super().update()

    def save(self):
        state=super().save()
        state['sit_critic_warm_updates']=self.critic_warm_updates
        state['sit_critic_warm_optimizer']=self.critic_warm_optimizer.state_dict()
        return state

    def load(self,loaded_dict,load_cfg,strict):
        result=super().load(loaded_dict,load_cfg,strict)
        self.critic_warm_updates=loaded_dict.get('sit_critic_warm_updates',0)
        if 'sit_critic_warm_optimizer' in loaded_dict:
            self.critic_warm_optimizer.load_state_dict(loaded_dict['sit_critic_warm_optimizer'])
        return result
