import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_hd1910_sit_head_env_cfg import head_contact_cost
from mjlab_microduck.tasks.getup_mechanical_contract import LIMITS

def test_stand_only_ground_cost():
    found=torch.tensor([[[True]],[[True]],[[False]]]);force=torch.ones(3,1,1,3)*8
    result=head_contact_cost(found,force,torch.tensor([1.,0.,1.]))
    assert result[0]>1 and result[1]==0 and result[2]==0

def test_control_only_differs_in_head_cost():
    a=load_env_cfg('Mjlab-SitHeadControl-Flat-MicroDuck-HD1910');b=load_env_cfg('Mjlab-SitHeadRepair-Flat-MicroDuck-HD1910')
    assert a.actions['joint_pos'].alpha==b.actions['joint_pos'].alpha==0
    assert a.actions['joint_pos'].clip==b.actions['joint_pos'].clip==LIMITS
    assert a.rewards['standing_head_ground'].weight==0 and b.rewards['standing_head_ground'].weight==-1
    assert repr(a.scene.sensors)==repr(b.scene.sensors)
    a.rewards['standing_head_ground'].weight=-1.
    assert repr(a.rewards)==repr(b.rewards)
