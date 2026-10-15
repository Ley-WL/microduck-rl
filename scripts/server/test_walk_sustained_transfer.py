"""Sustained command windows, physical invariants, and reward sign checks."""
import torch
from types import SimpleNamespace
from mjlab_microduck.tasks.walk_sustained_transfer import make_cfg, rl_cfg, SustainedCommand, SustainedBias

cfg=make_cfg()
assert cfg.decimation*cfg.sim.mujoco.timestep==.02
assert cfg.scene.entities['robot'].articulation.actuators[0].kp_fw==5.
assert cfg.rewards['long_stride'].params['complete_swing']
assert cfg.rewards['long_stride'].params['require_alternation']
assert cfg.rewards['tracking_bias'].weight<0
assert cfg.rewards['stop_motion'].weight<0
assert rl_cfg().algorithm.learning_rate==1e-5
cmd=SimpleNamespace(stop_time=torch.zeros(3),stop_speed=torch.tensor([.2,-.2,0.]),
    stop_yaw=torch.tensor([.6,-.6,0.]),vel_command_b=torch.zeros(3,3),
    is_standing_env=torch.zeros(3,dtype=torch.bool),is_heading_env=torch.ones(3,dtype=torch.bool),
    is_world_env=torch.ones(3,dtype=torch.bool))
for t,d in [(0,0),(2,1),(10,1),(12,0),(17,-1),(25,-1),(28,0),(32,1)]:
    cmd.stop_time[:]=t;SustainedCommand.compute(cmd,0.)
    assert torch.allclose(cmd.vel_command_b[:,0],cmd.stop_speed*d)
    assert torch.allclose(cmd.vel_command_b[:,2],cmd.stop_yaw*d)
    assert bool(cmd.is_standing_env[2])
data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(3,3),root_link_ang_vel_b=torch.zeros(3,3))
env=SimpleNamespace(num_envs=3,device='cpu',step_dt=.02,scene={'robot':SimpleNamespace(data=data)},
    command_manager=SimpleNamespace(get_command=lambda _:torch.zeros(3,3)))
term=SustainedBias(None,env)
assert torch.equal(term(env),torch.zeros(3))
data.root_link_ang_vel_b[:,2]=.5
for _ in range(100):cost=term(env)
assert bool((cost>0).all()) and bool((cost<=25).all())
term.reset(torch.tensor([0]));assert bool((term.error[0]==0).all())
assert bool((term.error[1]!=0).any())
print('sustained command/reward/partial reset/PD/stride invariants passed')
