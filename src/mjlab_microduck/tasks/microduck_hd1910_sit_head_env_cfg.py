"""Isolated current no-EMA sitstand contract; never changes recovery tasks."""
from copy import deepcopy
import torch
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactMatch,ContactSensorCfg
from .getup_mechanical_contract import LIMITS,edit_mechanical_spec
from .microduck_hd1910_getup_mechanical_env_cfg import MechanicalGetupActionCfg
from .microduck_hd1910_skills_env_cfg import make_hd1910_sitstand_env_cfg,get_skills_spec,HD1910SitStandRlCfg

def head_contact_cost(found,force,stand):
    hit=found.reshape(found.shape[0],-1).any(-1).float()
    magnitude=force.reshape(force.shape[0],-1,3).norm(dim=-1).sum(-1)
    # Soft cost, not a hard physical restriction; do not forbid body support in getup.
    return stand*hit*(1.+.05*(magnitude/7.23).clamp(max=4.).square())

def standing_head_ground_cost(env,sensor_name='head_assembly_ground'):
    data=env.scene.sensors[sensor_name].data
    stand=1.-env.command_manager.get_command('twist')[:,0].clamp(0.,1.)
    return head_contact_cost(data.found,data.force,stand)

def get_sit_mechanical_spec():return edit_mechanical_spec(get_skills_spec())

def make_sit_head_env_cfg(play=False,penalty=True):
    cfg=make_hd1910_sitstand_env_cfg(play)
    cfg.scene.entities['robot'].spec_fn=get_sit_mechanical_spec
    cfg.actions['joint_pos']=MechanicalGetupActionCfg(**vars(cfg.actions['joint_pos']))
    cfg.actions['joint_pos'].alpha=0.
    cfg.actions['joint_pos'].clip=dict(LIMITS)
    sensor=ContactSensorCfg(name='head_assembly_ground',primary=ContactMatch(mode='subtree',pattern='neck_pitch',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1)
    cfg.scene.sensors=tuple(cfg.scene.sensors)+(sensor,)
    cfg.rewards['standing_head_ground']=RewardTermCfg(func=standing_head_ground_cost,weight=-1. if penalty else 0.)
    return cfg

SitHeadRlCfg=deepcopy(HD1910SitStandRlCfg)
SitHeadRlCfg.experiment_name='hd1910_sit_head_mechanical'
SitHeadRlCfg.algorithm.learning_rate=1e-5
SitHeadRlCfg.max_iterations=20
SitHeadRlCfg.save_interval=5
