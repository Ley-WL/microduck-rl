"""GPU feedback references with whole-world split; privileged teacher only."""
import hashlib,json
from pathlib import Path
from dataclasses import asdict
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_sustained_transfer import make_cfg,rl_cfg
from walk_feedback_teacher import TensorFeedbackTeacher

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
torch.manual_seed(211131);cfg=make_cfg();cfg.scene.num_envs=100
cfg.observations['actor'].enable_corruption=False
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    runner.load(str(ROOT/'inputs/seed.pt'),load_cfg={'actor':True},map_location=env.device)
    actor=runner.alg.actor;actor.eval();obs,_=env.reset(seed=211131)
    cases=[(v,y) for v in (-.2,-.1,0.,.1,.2) for y in (-.6,-.3,0.,.3,.6)]
    grid=torch.tensor(cases,device=env.device).repeat_interleave(4,dim=0)
    term=env.command_manager.get_term('twist');term.stop_speed[:]=grid[:,0];term.stop_yaw[:]=grid[:,1]
    robot=env.scene['robot'];teacher=TensorFeedbackTeacher()
    failed=torch.zeros(100,dtype=torch.bool,device=env.device);peak=torch.zeros(100,device=env.device)
    xs=[];ys=[];commands=[];actuals=[]
    for tick in range(1500):
        measured=torch.stack((robot.data.root_link_lin_vel_b[:,0],robot.data.root_link_ang_vel_b[:,2]),-1)
        active=env.command_manager.get_command('twist').norm(dim=-1)>.001
        view=obs['actor'].clone();view[:,48:51]=teacher.query(view[:,48:51],measured,enabled=active)
        with torch.inference_mode():action=actor({**obs,'actor':view})
        assert bool(torch.isfinite(action).all())
        xs.append(obs['actor'].detach().cpu());ys.append(action.detach().cpu())
        commands.append(obs['actor'][:,[48,50]].detach().cpu());actuals.append(measured.detach().cpu())
        obs,_,done,timeout,_=env.step(action)
        tilt=torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1.,1.)))
        peak=torch.maximum(peak,torch.where(~failed,tilt,0.));failed|=done|timeout|(tilt>=45.)
        if tick%100==0:(OUT/'progress.json').write_text(json.dumps(dict(stage='collect',tick=tick,worlds=100,failed=int(failed.sum()))))
    survived=(~failed).cpu()
    torch.save(dict(observation=torch.stack(xs),target=torch.stack(ys),survived=survived,
        validationWorlds=(torch.arange(100)%4==3)&survived,cases=cases,
        command=torch.stack(commands),actual=torch.stack(actuals)),OUT/'gpu-feedback-data.pt')
    audit=dict(worlds=100,survived=int(survived.sum()),maxTiltDeg=float(peak.max()),seed=211131,
        steps=1500,trainingWorlds=75,validationWorlds=25,privilegedTeacherOnly=True,
        dataSha256=hashlib.sha256((OUT/'gpu-feedback-data.pt').read_bytes()).hexdigest())
    (OUT/'gpu-feedback-audit.json').write_text(json.dumps(audit,indent=2))
    assert bool(survived.all()),'Inspect failed reference worlds before any learning'
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',training=False,hardwareAccess=False)))
    print(audit,flush=True)
finally:env.close()
