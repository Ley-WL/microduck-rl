"""Donor ONNX replay with its runtime EMA and raw previous-action observation.

Uses evaluate_walk metrics; select donor or replica task explicitly. No hardware.
"""
from pathlib import Path

source = Path(__file__).with_name('evaluate_walk.py').read_text(encoding='utf-8')
source = source.replace("p.add_argument('--video');a=p.parse_args()", """p.add_argument('--balance-push',type=float,default=0.,help='Diagnostic world XY velocity impulse at 5s and 10s, m/s')
p.add_argument('--video');a=p.parse_args()""")
source = source.replace("t=tick*env.step_dt", """t=tick*env.step_dt
        if a.balance_push and tick in (round(5/env.step_dt),round(10/env.step_dt)):
            velocity=robot.data.root_link_vel_w.clone()
            directions=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],device=env.device)
            offsets=directions[torch.arange(a.envs,device=env.device)%4]*a.balance_push
            if tick==round(10/env.step_dt): offsets=-offsets
            velocity[:,:2]+=offsets
            robot.write_root_link_velocity_to_sim(velocity)""")
source = source.replace("env=ManagerBasedRlEnv(cfg,device='cuda:0')", """cfg.events.pop('push_robot',None)
for name in ('head_pose', 'body_pose'):
    cfg.commands[name].ranges=tuple((0.,0.) for _ in cfg.commands[name].ranges)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
filtered=np.zeros((a.envs,14),dtype=np.float32)""")
source = source.replace("obs,reward,terminated,truncated,_=env.step(torch.tensor(action,device=env.device))", """filtered[:]=.45*filtered+.55*action
        obs,reward,terminated,truncated,_=env.step(torch.tensor(filtered,device=env.device))
        # Runtime feeds the raw policy output back, rather than its EMA target.
        obs['actor'][:,34:48]=torch.tensor(action,device=env.device)
        reset_ids=(terminated|truncated).cpu().numpy()
        filtered[reset_ids]=0.
        obs['actor'][terminated|truncated,34:48]=0.""")
source = source.replace("result['gait']=gait", "result['runtimeActionEmaOldWeight']=.45\n    result['rawPreviousAction']=True\n    result['externalPushes']=False\n    result['balanceVelocityImpulseMps']=a.balance_push\n    result['maxTiltDeg']=float(np.max(tilts))\n    result['gait']=gait")
exec(compile(source,str(Path(__file__).with_name('evaluate_walk.py')),'exec'))
