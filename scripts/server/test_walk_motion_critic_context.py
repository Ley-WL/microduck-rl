"""Critic-only context must expose memory without mutating it or the actor contract."""
from types import SimpleNamespace
import torch
from mjlab_microduck.tasks import walk_mean_motion_tracking as old
from mjlab_microduck.tasks import walk_motion_critic_context as new

a,b=old.make_cfg(),new.make_cfg()
for field in ('sim','decimation','actions','commands','events','terminations','rewards'):
    assert getattr(a,field)==getattr(b,field),field
assert a.observations['actor']==b.observations['actor']
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
assert set(b.observations['critic'].terms)==set(a.observations['critic'].terms)|{'reward_context'}
for name,term in a.observations['critic'].terms.items():assert term==b.observations['critic'].terms[name]
plan=SimpleNamespace(stop_time=torch.tensor([0.,15.,30.]),stop_speed=torch.tensor([.2,-.05,.2]),
    stop_yaw=torch.tensor([.6,0.,.6]))
env=SimpleNamespace(num_envs=3,device='cpu',command_manager=SimpleNamespace(get_term=lambda _:plan))
initial=new.reward_context(env);assert initial.shape==(3,10) and not initial[:,:6].any()
torch.testing.assert_close(initial[0,6:],initial[2,6:],atol=1e-6,rtol=0)
linear=SimpleNamespace(error=torch.arange(9,dtype=torch.float32).reshape(3,3))
angular=SimpleNamespace(error=-linear.error.clone())
env.reward_manager=SimpleNamespace(get_term_cfg=lambda name:SimpleNamespace(func=
    linear if name=='track_linear_velocity' else angular))
before=linear.error.clone();features=new.reward_context(env)
assert torch.equal(features[:,:3],linear.error) and torch.equal(features[:,3:6],angular.error)
assert torch.equal(linear.error,before),'Reading observations cannot update reward memory'
linear.error[1]=0.;angular.error[1]=0.;plan.stop_time[1]=0.;plan.stop_speed[1]=0.;plan.stop_yaw[1]=0.
reset=new.reward_context(env);assert not reset[1,:6].any()
assert torch.equal(reset[[0,2]],features[[0,2]])
assert not torch.equal(features[1],reset[1]),'Same physical state can have different reward memory/plan'
print('Actor/config parity; critic-only 10D reward memory and command phase; construction/reset/read-only checks passed')
