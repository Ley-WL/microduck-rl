"""Warm-start getup under current physical stops AND runtime output bounds."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from . import mdp
from .getup_mechanical_contract import NAMES,LIMITS,REFERENCES,DIRECTIONS,ENCODER_STEP,edit_mechanical_spec
from .microduck_hd1910_skills_env_cfg import make_hd1910_getup_env_cfg,get_skills_spec,HD1910GetupRlCfg
from .xgoduck_runtime_actions import XgoDuckPositionAction,XgoDuckPositionActionCfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
import re

@dataclass(kw_only=True)
class MechanicalGetupActionCfg(XgoDuckPositionActionCfg):
    def build(self,env):
        return MechanicalGetupAction(self,env)

class MechanicalGetupAction(XgoDuckPositionAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert tuple(self._target_names)==NAMES
        assert cfg.scale==1., 'The board getup contract uses scale 1'
        self._filtered_actions=self._filtered_actions.double()
        self._home64=torch.tensor([next(v for k,v in HOME_FRAME.joint_pos.items() if re.fullmatch(k,n)) for n in NAMES],device=self.device,dtype=torch.float64)
        torch.testing.assert_close(self._offset.double(),self._home64.expand_as(self._offset),atol=3e-8,rtol=0)
        self._bounds64=torch.tensor(list(LIMITS.values()),device=self.device,dtype=torch.float64)
        self._references=torch.tensor(REFERENCES,device=self.device,dtype=torch.float64)
        self._directions=torch.tensor(DIRECTIONS,device=self.device,dtype=torch.float64)

    def process_actions(self,actions):
        # Rust uses f64 EMA/HOME/encoder arithmetic. Keep f64 through rounding,
        # then hand float32 targets to BAM.
        self._filtered_actions.mul_(self.cfg.alpha).add_(actions.double(),alpha=1-self.cfg.alpha)
        self._raw_actions.copy_(actions)
        target=(self._filtered_actions+self._home64).clamp(self._bounds64[:,0],self._bounds64[:,1])
        encoded=torch.floor(self._references+target/ENCODER_STEP*self._directions+.5)
        self._processed_actions=((encoded-self._references)*ENCODER_STEP*self._directions).to(self._raw_actions.dtype)

    @property
    def position_target(self):
        return self._processed_actions

def get_mechanical_spec():
    return edit_mechanical_spec(get_skills_spec())

def head_max_bias(env,**params):
    # Preserve the source getup4750 v3 terminal head objective on both roots.
    mean_reward=mdp.head_pose_bias_penalty(env,**params)
    magnitude=env._head_bias_ema.abs()
    return mean_reward*(magnitude.amax(-1)/magnitude.mean(-1).clamp_min(1e-12))

def make_getup_mechanical_env_cfg(play=False):
    cfg=make_hd1910_getup_env_cfg(play=play)
    cfg.scene.entities['robot'].spec_fn=get_mechanical_spec
    cfg.actions['joint_pos']=MechanicalGetupActionCfg(**vars(cfg.actions['joint_pos']))
    cfg.actions['joint_pos'].clip=dict(LIMITS)
    cfg.rewards['head_pose_bias'].func=head_max_bias
    cfg.rewards['head_pose_bias'].weight=1.5
    cfg.curriculum.pop('head_pose_bias_weight',None)
    # Old late-stage fallen coverage from iteration zero; do not restart easy
    # upright-only curriculum just because optimizer iteration is reset.
    cfg.curriculum.pop('recovery_pose_mix',None)
    cfg.events['set_ground_state'].params['pose_probabilities']=(.025,.025,.16,.2,.18,.18,.115,.115)
    return cfg

GetupMechanicalRlCfg=deepcopy(HD1910GetupRlCfg)
GetupMechanicalRlCfg.experiment_name='hd1910_getup_mechanical'
GetupMechanicalRlCfg.run_name='mechanical-20261012'
GetupMechanicalRlCfg.algorithm.learning_rate=3e-4
GetupMechanicalRlCfg.max_iterations=300
GetupMechanicalRlCfg.save_interval=100
