"""Paired frozen stop-cycle screen; never counts recovery after a reset as success."""
import json,hashlib
from pathlib import Path
from dataclasses import asdict
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_stop_transfer import make_cfg,rl_cfg
ROOT=Path(__file__).resolve().parents[2];rows=[]
for seed in [151031,151037]:
 starts=None
 for label,checkpoint in [('baseline','inputs/seed.pt'),('candidate','evidence/main/final.pt')]:
  torch.manual_seed(seed);cfg=make_cfg();cfg.scene.num_envs=64;cfg.episode_length_s=26
  cfg.observations['actor'].enable_corruption=False
  env=ManagerBasedRlEnv(cfg,device='cuda:0')
  try:
   agent=asdict(rl_cfg());agent['upload_model']=False
   runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
   runner.load(str(ROOT/checkpoint),load_cfg={'actor':True},map_location=env.device)
   actor=runner.alg.actor;actor.eval();obs,_=env.reset(seed=seed);robot=env.scene['robot']
   initial=torch.cat([robot.data.root_link_pos_w,robot.data.root_link_quat_w,robot.data.joint_pos,robot.data.joint_vel],dim=-1).cpu()
   if starts is None:starts=initial.clone()
   else:assert torch.equal(starts,initial),'Initial states not paired'
   failed=torch.zeros(64,dtype=torch.bool,device=env.device);peak=torch.zeros(64,device=env.device)
   stopvel=torch.zeros_like(peak);stopcount=torch.zeros_like(peak);moving=torch.zeros_like(peak);movecount=torch.zeros_like(peak)
   for tick in range(1200):
    with torch.inference_mode():action=actor(obs)
    obs,_,done,timeout,_=env.step(action)
    tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)))
    peak=torch.maximum(peak,torch.where(~failed,tilt,0.));failed|=done|timeout|(tilt>=45)
    phase=((tick+1)*.02)%12
    live=~failed
    if 4.3<=phase<7 or 10.3<=phase<12:
     stopvel+=torch.where(live,robot.data.root_link_lin_vel_b[:,:2].norm(dim=-1),0.);stopcount+=live
    if 2<=phase<4 or 8<=phase<10:
     desired=env.command_manager.get_command('twist')[:,0]
     moving+=torch.where(live,robot.data.root_link_lin_vel_b[:,0]*torch.sign(desired),0.);movecount+=live
   row=dict(policy=label,seed=seed,survival=(~failed).cpu().tolist(),maxTiltDeg=peak.cpu().tolist(),stopMeanSpeed=(stopvel/stopcount.clamp_min(1)).cpu().tolist(),movingSignedSpeed=(moving/movecount.clamp_min(1)).cpu().tolist(),initialStateSha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest())
   rows.append(row);print(label,seed,'survived',int((~failed).sum()),flush=True)
   (ROOT/'evidence/paired-screen.json').write_text(json.dumps(dict(results=rows,accepted=False,simulationOnly=True),indent=2))
  finally:env.close()
comparisons=[]
for seed in [151031,151037]:
 b,c=[r for r in rows if r['seed']==seed];paired=[i for i in range(64) if b['survival'][i] and c['survival'][i]]
 mean=lambda r,key:sum(r[key][i] for i in paired)/max(1,len(paired))
 lost=[i for i in range(64) if b['survival'][i] and not c['survival'][i]]
 comparisons.append(dict(seed=seed,lost=lost,pairedSurvivors=len(paired),baselineStopSpeed=mean(b,'stopMeanSpeed'),candidateStopSpeed=mean(c,'stopMeanSpeed'),baselineMovingSpeed=mean(b,'movingSignedSpeed'),candidateMovingSpeed=mean(c,'movingSignedSpeed')))
accepted=all(not r['lost'] and r['pairedSurvivors']>=60 and r['candidateStopSpeed']<=.95*r['baselineStopSpeed'] and r['candidateMovingSpeed']>=.9*r['baselineMovingSpeed'] for r in comparisons)
(ROOT/'evidence/paired-screen.json').write_text(json.dumps(dict(results=rows,comparison=comparisons,accepted=accepted,simulationOnly=True,hardwareValidated=False),indent=2))
print('SCREEN',accepted,comparisons,flush=True)
