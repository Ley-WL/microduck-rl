"""Bounded imitation of offline pulse demonstrations; actor has no clock input."""
import copy,hashlib,json,subprocess,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg,rl_cfg
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
torch.manual_seed(243131)
cfg=make_cfg();cfg.scene.num_envs=64
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    runner.load(str(ROOT/'inputs/seed.pt'),load_cfg={'actor':True},map_location=env.device)
    actor=runner.alg.actor.eval();actor.update_normalization=lambda *a,**k:None
    for p in actor.distribution.parameters():p.requires_grad_(False)
    norms={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    def predict(x):return actor(TensorDict({'actor':x},batch_size=[len(x)]))
    data_path=ROOT/'inputs/demonstration.npz';proof=json.loads((ROOT/'inputs/demo-audit.json').read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==proof['dataSha256']
    data=np.load(data_path);mask=data['validation'];world=data['world']
    assert len(set(world[~mask]))>=2 and len(set(world[mask]))>=2
    assert set(world[~mask]).isdisjoint(set(world[mask]))
    assert all(np.all(mask[world==w]==mask[world==w][0]) for w in np.unique(world))
    x,y,vx,vy=[torch.tensor(a,device=env.device) for a in (data['observation'][~mask],data['target'][~mask],data['observation'][mask],data['target'][mask])]
    path=ROOT/'inputs/retention.npz';retention=json.loads((ROOT/'inputs/retention-audit.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==retention['sha256']
    ax=torch.tensor(np.load(path)['observation'],device=env.device)
    anchor=copy.deepcopy(actor).eval()
    anchor_path=ROOT/'inputs/anchor.pt'
    if anchor_path.exists():anchor.load_state_dict(torch.load(anchor_path,map_location=env.device,weights_only=False)['actor_state_dict'])
    with torch.no_grad():ay=torch.cat([anchor(TensorDict({'actor':z},batch_size=[len(z)])) for z in ax.split(1024)])
    del anchor
    screen_seeds=json.loads((ROOT/'inputs/screen-seeds.json').read_text()) if (ROOT/'inputs/screen-seeds.json').exists() else [243171,243177]
    screen_speeds=json.loads((ROOT/'inputs/screen-speeds.json').read_text()) if (ROOT/'inputs/screen-speeds.json').exists() else [-.05,.05]
    stop_data=None
    if (ROOT/'inputs/stop-data.npz').exists():
        stop_data=np.load(ROOT/'inputs/stop-data.npz');sm=stop_data['validation'];sw=stop_data['world']
        assert set(sw[sm]).isdisjoint(set(sw[~sm]))
        sx,sy,svx,svy=[torch.tensor(z,device=env.device) for z in (stop_data['observation'][~sm],stop_data['target'][~sm],stop_data['observation'][sm],stop_data['target'][sm])]
    optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-4)
    def error():
        with torch.no_grad():return float(torch.cat([(predict(z)-t).square() for z,t in zip(vx.split(1024),vy.split(1024))]).mean().sqrt())
    initial=error();previous=0;records=[]
    with torch.no_grad():initial_stop=float((predict(svx)-svy).square().mean().sqrt()) if stop_data is not None else None
    baseline=OUT/'baseline';baseline.mkdir()
    with (baseline/'cpu.log').open('w') as log:
        subprocess.run([sys.executable,'scripts/evaluate_walk_runtime_cpu.py','--onnx',str(ROOT/'inputs/student.onnx'),'--physics-profile','kissqy','--output',str(baseline/'cpu-straight'),'--speeds',*[str(s) for s in screen_speeds],'--seeds',*[str(s) for s in screen_seeds],'--trials','1','--seconds','30','--transport-ms','0'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'scripts/analyze_walk_runtime.py',str(baseline),'--seconds','30','--window','17','27'],stdout=log,stderr=subprocess.STDOUT,check=True)
    for total in (5,100,500):
        (OUT/'fit-progress.json').write_text(json.dumps(dict(stage='fit',updates=total)))
        for _ in range(previous,total):
            ids=torch.randint(len(x),(512,),device=env.device);aid=torch.randint(len(ax),(512,),device=env.device)
            loss=(predict(x[ids])-y[ids]).square().mean()+(predict(ax[aid])-ay[aid]).square().mean()
            if stop_data is not None:
                sid=torch.randint(len(sx),(512,),device=env.device)
                loss=loss+(predict(sx[sid])-sy[sid]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in norms.items())
        assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
        stage=OUT/f'step{total}';stage.mkdir();runner.save(str(stage/'checkpoint.pt'));runner.export_policy_to_onnx(str(stage),'policy.onnx')
        torch.save(dict(optimizer=optimizer.state_dict(),updates=total),stage/'supervised-state.pt')
        session=ort.InferenceSession(str(stage/'policy.onnx'),providers=['CPUExecutionProvider']);assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
        with torch.no_grad():expected=predict(vx[:64]).cpu().numpy()
        actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:z[None]})[0] for z in vx[:64].cpu().numpy()]);parity=float(abs(actual-expected).max());assert parity<2e-5
        sha=hashlib.sha256((stage/'policy.onnx').read_bytes()).hexdigest();meta=json.loads((ROOT/'inputs/student.metadata.json').read_text());meta['policySha256']=sha;meta['walkExecution']['policySha256']=sha;meta['hardwareValidated']=False;meta['offlinePulseImitation']=dict(clockAtRuntime=False,accepted=False,updates=total);(stage/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
        rms=error();record=dict(updates=total,validationRms=rms,initialValidationRms=initial,normalizerUnchanged=True,finite=True,onnxMaxError=parity,policySha256=sha);records.append(record)
        if stop_data is not None:
            with torch.no_grad():record.update(stopValidationRms=float((predict(svx)-svy).square().mean().sqrt()),initialStopValidationRms=initial_stop)
        (OUT/'fit-audit.json').write_text(json.dumps(dict(records=records,hardwareAccess=False,accepted=False),indent=2))
        if total>5:
            with (stage/'cpu.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/evaluate_walk_runtime_cpu.py','--onnx',str(stage/'policy.onnx'),'--physics-profile','kissqy','--output',str(stage/'cpu-straight'),'--speeds',*[str(s) for s in screen_speeds],'--seeds',*[str(s) for s in screen_seeds],'--trials','1','--seconds','30','--transport-ms','0'],stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'scripts/analyze_walk_runtime.py',str(stage),'--seconds','30','--window','17','27'],stdout=log,stderr=subprocess.STDOUT,check=True)
            screen=json.loads((stage/'cpu-body-tracking.json').read_text());assert len(screen['results'])==len(screen_seeds)*len(screen_speeds)
            if not all(r['completed'] for r in screen['results']) or rms>=initial:break
        previous=total
finally:env.close()
