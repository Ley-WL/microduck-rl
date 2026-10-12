import math
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.sit_settled_tilt import settled_tilt_values


def test_tilt_cost_ignores_transitional_lean_but_rejects_tilted_endpoint():
    radians=torch.tensor([0.,10.,20.,35.,35.])*math.pi/180
    gravity=torch.stack([radians.sin(),torch.zeros(5),-radians.cos()],dim=-1)
    cost=settled_tilt_values(gravity,torch.zeros(5),torch.tensor([0.,0.,0.,0.,.5]))
    torch.testing.assert_close(cost,torch.tensor([0.,0.,.64,5.29,0.]))


def test_endpoint_constraint_keeps_action_observation_and_transition_rewards():
    old=load_env_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    new=load_env_cfg('Mjlab-SitStandProcessUpright-Flat-MicroDuck-HD1910')
    assert new.actions==old.actions and new.observations==old.observations
    assert new.commands==old.commands
    assert new.rewards['settled_tilt'].weight==-2.
    for name,term in old.rewards.items():assert new.rewards[name]==term


def test_capped_cost_keeps_mild_tilt_signal_and_limits_fallen_sample_weight():
    radians=torch.tensor([0.,20.,35.,90.,180.,90.])*math.pi/180
    gravity=torch.stack([radians.sin(),torch.zeros(6),-radians.cos()],dim=-1)
    cost=settled_tilt_values(gravity,torch.zeros(6),torch.tensor([0.,0.,0.,0.,0.,.5]),max_cost=4.)
    torch.testing.assert_close(cost,torch.tensor([0.,.64,4.,4.,4.,0.]))


def test_capped_case_changes_only_endpoint_cost_cap():
    old=load_env_cfg('Mjlab-SitStandProcessUpright-Flat-MicroDuck-HD1910')
    new=load_env_cfg('Mjlab-SitStandProcessUprightCapped-Flat-MicroDuck-HD1910')
    assert new.actions==old.actions and new.observations==old.observations
    assert new.commands==old.commands and new.events==old.events
    assert new.rewards['settled_tilt'].params=={'max_cost':4.}
    assert new.rewards['settled_tilt'].weight==old.rewards['settled_tilt'].weight
    for name,term in old.rewards.items():
        if name!='settled_tilt':assert new.rewards[name]==term
