"""Small config and schedule gate before any optimizer update."""
import torch
from types import SimpleNamespace
from mjlab_microduck.tasks.walk_stop_transfer import make_cfg,rl_cfg
from mjlab_microduck.tasks.mdp import walk_stop_schedule,walk_stop_motion_cost,walk_stop_tilt_cost
cfg=make_cfg();a=cfg.scene.entities['robot'].articulation.actuators[0]
assert a.kp_fw==5 and cfg.decimation*cfg.sim.mujoco.timestep==.02
assert cfg.rewards['stop_motion'].weight<0 and cfg.rewards['stop_tilt'].weight<0
assert rl_cfg().algorithm.schedule=='fixed'
t=torch.tensor([0.,1.,3.99,4.,6.99,7.,9.99,10.,11.99,12.])
c=walk_stop_schedule(t,torch.full_like(t,.2),torch.full_like(t,.6))
assert torch.allclose(c[:,0],torch.tensor([0,.2,.2,0,0,-.2,-.2,0,0,0]))
assert torch.allclose(c[:,2],c[:,0]*3)
data=SimpleNamespace(root_link_lin_vel_b=torch.ones(2,3)*.1,root_link_ang_vel_b=torch.ones(2,3),projected_gravity_b=torch.tensor([[0.,0.,-.9],[0.,0.,-.9]]))
env=SimpleNamespace(scene={'robot':SimpleNamespace(data=data)},command_manager=SimpleNamespace(get_command=lambda _:torch.tensor([[0.,0.,0.],[.2,0.,0.]])))
assert walk_stop_motion_cost(env)[0]>0 and walk_stop_motion_cost(env)[1]==0
assert walk_stop_tilt_cost(env)[0]>0 and walk_stop_tilt_cost(env)[1]==0
print('stop config/schedule/reward gates passed')
