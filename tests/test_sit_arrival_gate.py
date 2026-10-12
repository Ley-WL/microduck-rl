import math
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.sit_arrival_gate import arrival_gate_values


def test_gate_keeps_correct_endpoints_and_transitions_but_removes_tilted_rest_pay():
    angles=torch.tensor([0.,10.,16.,20.,35.,90.,35.])*math.pi/180
    quat=torch.stack([(angles/2).cos(),torch.zeros(7),(angles/2).sin(),torch.zeros(7)],dim=-1)
    gate=arrival_gate_values(quat,torch.zeros(7),torch.tensor([0.,0.,0.,0.,0.,0.,.5]))
    torch.testing.assert_close(gate[[0,1,3,4,5,6]],torch.tensor([1.,1.,0.,0.,0.,1.]),atol=1e-6,rtol=0)
    assert 0<gate[2]<1


def test_gate_preserves_rewards_weights_params_and_all_motion_interfaces():
    a=load_env_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    b=load_env_cfg('Mjlab-SitStandProcessArrivalGate-Flat-MicroDuck-HD1910')
    for field in ('actions','observations','commands','events','curriculum','terminations'):
        assert getattr(a,field)==getattr(b,field)
    changed={'posture_pose_legs','posture_height','posture_height_sharp','posture_stillness'}
    assert a.rewards.keys()==b.rewards.keys()
    for name in a.rewards:
        old,new=a.rewards[name],b.rewards[name]
        assert old.weight==new.weight and old.params==new.params
        assert (old.func!=new.func)==(name in changed)
