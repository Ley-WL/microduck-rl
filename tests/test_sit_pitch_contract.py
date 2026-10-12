import math
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_hd1910_sit_pitch_env_cfg import forward_pitch_cost

def test_forward_cost_before_contact_and_sit_exemption():
    gravity=torch.tensor([[0.,0.,-1.],[math.sin(math.radians(30)),0.,-math.cos(math.radians(30))],[-.5,0.,-.866],[.5,0.,-.866]])
    cost=forward_pitch_cost(gravity,torch.tensor([1.,1.,1.,0.]))
    assert cost[0]==0 and cost[1]>0 and cost[2]==0 and cost[3]==0
    assert torch.isfinite(cost).all()

def test_single_reward_difference():
    a=load_env_cfg('Mjlab-SitPitchControl-Flat-MicroDuck-HD1910');b=load_env_cfg('Mjlab-SitPitchRepair-Flat-MicroDuck-HD1910')
    assert a.rewards['standing_head_ground'].weight==b.rewards['standing_head_ground'].weight==-1.
    assert a.rewards['standing_forward_pitch'].weight==0 and b.rewards['standing_forward_pitch'].weight==-2.
    a.rewards['standing_forward_pitch'].weight=-2.
    assert repr(a.rewards)==repr(b.rewards) and repr(a.actions)==repr(b.actions)
    assert repr(a.observations)==repr(b.observations) and repr(a.commands)==repr(b.commands)

def test_fixed_lr_only_changes_scheduler_and_output_directory():
    from mjlab.tasks.registry import load_rl_cfg
    from dataclasses import asdict
    from copy import deepcopy
    a=deepcopy(load_rl_cfg('Mjlab-SitPitchRepair-Flat-MicroDuck-HD1910'));b=deepcopy(load_rl_cfg('Mjlab-SitPitchFixedRepair-Flat-MicroDuck-HD1910'))
    assert a.algorithm.schedule=='adaptive' and b.algorithm.schedule=='fixed'
    assert a.algorithm.learning_rate==b.algorithm.learning_rate==1e-5
    a.algorithm.schedule='fixed';a.experiment_name=b.experiment_name
    assert asdict(a)==asdict(b)
