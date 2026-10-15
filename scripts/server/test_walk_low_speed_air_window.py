"""Compare actual mjlab reward on synthetic contact times, including zero gates."""
from types import SimpleNamespace
import torch
from mjlab_microduck.tasks import walk_low_speed_air_window as candidate
from mjlab_microduck.tasks import walk_low_speed_focused_control as control

a,b=control.make_cfg(),candidate.make_cfg()
for field in ('sim','decimation','actions','observations','commands','events','terminations'):
    assert getattr(a,field)==getattr(b,field),field
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
for name in a.rewards:
    if name!='air_time':assert a.rewards[name]==b.rewards[name],name
assert a.rewards['air_time'].weight==b.rewards['air_time'].weight==3.
assert b.rewards['air_time'].params['original'] is a.rewards['air_time'].func
speeds=torch.tensor([0.,.02,.03,.05,-.05,.075,.1,.2,-.4])
commands=torch.zeros((len(speeds),3));commands[:,0]=speeds;commands[0,2]=.6
limits=candidate.air_time_limit(commands)
torch.testing.assert_close(limits,torch.tensor([.3,.3,.3,.6,.6,.4,.3,.3,.3]))
env=SimpleNamespace(scene={},command_manager=SimpleNamespace(get_command=lambda _:commands),extras={'log':{}})
original=a.rewards['air_time'];modified=b.rewards['air_time']
for tick in range(61):
    times=torch.full((len(speeds),2),tick*.02)
    env.scene['feet_ground_contact']=SimpleNamespace(data=SimpleNamespace(current_air_time=times))
    old=original.func(env,**original.params);new=modified.func(env,**modified.params)
    same=limits==.3
    assert torch.equal(old[same],new[same])
    assert torch.all(new>=old)
    if tick*.02>=.8:assert not new.any(),'Long holds must stop earning'
commands.zero_()
assert not modified.func(env,**modified.params).any()
env.scene['feet_ground_contact'].data.current_air_time.fill_(.2)
assert not modified.func(env,**modified.params).any(),'Zero commands must not earn'
print('Actual reward parity outside low speed, bounded time credit, zero command and config contrast passed')
