"""Actual reward oracle: steady errors penalized, oscillation averaged, stop unchanged."""
from types import SimpleNamespace
import math
import torch
from mjlab_microduck.tasks import walk_low_speed_mean_tracking as candidate
from mjlab_microduck.tasks import walk_low_speed_focused_control as control

a,b=control.make_cfg(),candidate.make_cfg()
for field in ('sim','decimation','actions','observations','commands','events','terminations'):
    assert getattr(a,field)==getattr(b,field),field
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
for name in a.rewards:
    if name not in ('track_linear_velocity','track_angular_velocity'):assert a.rewards[name]==b.rewards[name],name
command=torch.tensor([[.05,0,0],[.05,0,0],[.05,0,0],[.2,0,.6],[0,0,0],[0,0,.6]])
data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(6,3),root_link_ang_vel_b=torch.zeros(6,3))
env=SimpleNamespace(num_envs=6,device='cpu',step_dt=.02,
    scene={'robot':SimpleNamespace(data=data)},command_manager=SimpleNamespace(get_command=lambda _:command))
for name,angular in [('track_linear_velocity',False),('track_angular_velocity',True)]:
    term=b.rewards[name];tracker=candidate.MeanTracking(term,env);old=a.rewards[name]
    scores=[];legacy=[]
    for tick in range(1000):
        wave=math.sin(2*math.pi*4*tick*.02)
        actual=data.root_link_ang_vel_b if angular else data.root_link_lin_vel_b
        actual.zero_()
        if angular:
            actual[0,2]=wave;actual[2,2]=.6
        else:
            actual[0,0]=.05+.12*wave;actual[0,2]=.05*wave;actual[2,0]=-.05
        actual[3]=torch.tensor([.1,.2,.3]);actual[4]=torch.tensor([.2,.1,.4]);actual[5]=torch.tensor([.1,.3,.5])
        score=tracker(env,**term.params);original=old.func(env,**old.params)
        assert torch.equal(score[3:],original[3:])
        if tick>500:scores.append(score.clone());legacy.append(original.clone())
    mean=torch.stack(scores).mean(0);before=torch.stack(legacy).mean(0)
    assert mean[0]>before[0]+.3,'Accurate oscillating motion should not be scored as a DC error'
    assert mean[0]>mean[2]+.3,'Wrong mean direction must remain penalized'
    if not angular:assert mean[0]>mean[1]+.1,'Standing at a movement command remains worse'
    saved=tracker.error[1:].clone();tracker.reset(torch.tensor([0]))
    assert not tracker.error[0].any();assert torch.equal(saved,tracker.error[1:])
    print(name,'new',mean.tolist(),'legacy',before.tolist())
print('Non-low-speed parity, zero-command preservation, drift penalty, oscillation and partial reset passed')
