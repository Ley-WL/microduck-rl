"""Real mjlab reward checks: prefer accurate oscillating motion to slowing down."""
from types import SimpleNamespace
import math
import torch
from mjlab_microduck.tasks import walk_low_speed_mean_tracking as old
from mjlab_microduck.tasks import walk_mean_motion_tracking as new

a,b=old.make_cfg(),new.make_cfg()
for field in ('sim','decimation','actions','observations','commands','events','terminations'):
    assert getattr(a,field)==getattr(b,field),field
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
for name in a.rewards:
    if name not in ('track_linear_velocity','track_angular_velocity'):
        assert a.rewards[name]==b.rewards[name],name
command=torch.tensor([[.05,0,0],[.2,0,.6],[.2,0,.6],[.2,0,.6],[0,0,0],[0,0,.6]])
data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(6,3),root_link_ang_vel_b=torch.zeros(6,3))
env=SimpleNamespace(num_envs=6,device='cpu',step_dt=.02,scene={'robot':SimpleNamespace(data=data)},
    command_manager=SimpleNamespace(get_command=lambda _:command))
for name,angular in [('track_linear_velocity',False),('track_angular_velocity',True)]:
    term=b.rewards[name];before=old.MeanTracking(a.rewards[name],env);after=new.MeanMotionTracking(term,env)
    old_scores=[];new_scores=[]
    for tick in range(1000):
        wave=math.sin(2*math.pi*4*tick*.02)
        actual=data.root_link_ang_vel_b if angular else data.root_link_lin_vel_b
        actual.zero_()
        if angular:
            actual[0,2]=wave;actual[1,2]=.6+wave;actual[2,2]=.4;actual[3,2]=-.6
            actual[5,2]=.6+wave
        else:
            actual[0,0]=.05+.12*wave
            actual[1]=torch.tensor([.2+.12*wave,0,.05*wave])
            actual[2,0]=.15;actual[3,0]=-.2
        actual[4]=torch.tensor([.2,.1,.4])
        legacy=before(env,**term.params);score=after(env,**term.params)
        assert torch.equal(after.legacy_reward,legacy)
        assert torch.equal(score[[0,4]],legacy[[0,4]]),'Low-speed and stop parity'
        if tick>500:old_scores.append(legacy.clone());new_scores.append(score.clone())
    previous=torch.stack(old_scores).mean(0);current=torch.stack(new_scores).mean(0)
    assert previous[1]<previous[2],'Instantaneous reference must expose the slowdown preference'
    assert current[1]>current[2]+.1,'Correct mean motion must beat slow motion'
    assert current[1]>current[3]+.3,'Wrong direction must be penalized'
    if angular:assert current[5]>previous[5]+.3,'Pure turn oscillation must average'
    saved=after.error[1:].clone();after.reset(torch.tensor([0]))
    assert not after.error[0].any() and torch.equal(saved,after.error[1:])
    print(name,'old',previous.tolist(),'new',current.tolist())
print('Motion reward ordering, exact legacy/low-speed/stop parity, config and partial reset passed')
