import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.sit_state_reference import posture_reference, StateReferenceCommand


def test_reference_is_state_based_bounded_and_bidirectional():
    home=torch.zeros(4,4)
    seated=torch.tensor([[-.45,1.25,.45,-1.25]]).repeat(4,1)
    position=home+(seated-home)*torch.tensor([0.,.4,.6,1.])[:,None]
    flag=torch.tensor([1.,1.,0.,0.])
    expected=torch.tensor([.1,.5,.5,.9])
    torch.testing.assert_close(posture_reference(position,home,seated,flag),expected)
    torch.testing.assert_close(posture_reference(home,home,seated,torch.zeros(4)),torch.zeros(4))
    torch.testing.assert_close(posture_reference(seated,home,seated,torch.ones(4)),torch.ones(4))


def test_only_reward_reference_changes_from_weak_process():
    old=load_env_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    new=load_env_cfg('Mjlab-SitStandProcessStateReference-Flat-MicroDuck-HD1910')
    assert new.commands['twist'].class_type is StateReferenceCommand
    assert old.commands['twist'].class_type is not StateReferenceCommand
    assert new.actions['joint_pos']==old.actions['joint_pos']
    assert new.observations==old.observations
    assert new.rewards==old.rewards
