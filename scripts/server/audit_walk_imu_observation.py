"""Check actor IMU slots against free-joint coordinates, without hardware."""
import json
from pathlib import Path
from dataclasses import asdict
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg, rl_cfg

root=Path(__file__).resolve().parents[2]
cfg=make_cfg();cfg.scene.num_envs=8
cfg.observations['actor'].enable_corruption=False
for term in cfg.observations['actor'].terms.values():
    if 'max_angle_deg' in term.params:term.params['max_angle_deg']=0.
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    runner.load(str(root/'inputs/seed.pt'),load_cfg={'actor':True},map_location=env.device)
    actor=runner.alg.actor;actor.eval();obs,_=env.reset(seed=205101)
    robot=env.scene['robot'];model=env.sim.mj_model
    joint=model.joint('robot/trunk_base_freejoint');qadr=int(joint.qposadr[0]);vadr=int(joint.dofadr[0])
    errors=[]
    for tick in range(160):
        q=env.sim.data.qpos[:,qadr+3:qadr+7]
        w,x,y,z=q.unbind(-1)
        expected_gravity=torch.stack((2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1),-1)
        expected_gyro=env.sim.data.qvel[:,vadr+3:vadr+6]
        errors.append([float((obs['actor'][:,:3]-expected_gyro).abs().max()),
            float((obs['actor'][:,3:6]-expected_gravity).abs().max())])
        with torch.inference_mode():action=actor(obs)
        obs,*_=env.step(action)
    result=dict(maxGyroError=max(x[0] for x in errors),maxGravityError=max(x[1] for x in errors),
        observationScales={n:t.scale for n,t in cfg.observations['actor'].terms.items()})
    (root/'evidence/imu-observation-audit.json').write_text(json.dumps(result,indent=2))
    print(result)
finally:env.close()
