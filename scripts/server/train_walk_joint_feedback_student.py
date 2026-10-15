"""Joint CPU/GPU feedback distillation; standalone actor with frozen normalization."""
import hashlib,json,subprocess,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg,rl_cfg

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence'
torch.manual_seed(212131)
options_path=ROOT/'inputs/training-options.json'
options=json.loads(options_path.read_text()) if options_path.exists() else {}
stop_weight=float(options.get('stopWeight',1.))
assert 1 <= stop_weight <= 16
def imitation_loss(observation,prediction,target):
    weight=torch.where(observation[:,48:51].norm(dim=-1)<.02,stop_weight,1.)
    return ((prediction-target).square().mean(-1)*weight).sum()/weight.sum()
data_audit=json.loads((ROOT/'inputs/data-audit.json').read_text())
assert hashlib.sha256((ROOT/'inputs/feedback-data.npz').read_bytes()).hexdigest()==data_audit['dataSha256']
assert len(data_audit['worlds'])==75 and all(w['valid'] for w in data_audit['worlds'])
cfg=make_cfg();cfg.scene.num_envs=64
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    agent=asdict(rl_cfg());agent['upload_model']=False
    runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env),agent,log_dir=None,device=env.device)
    runner.load(str(ROOT/'inputs/seed.pt'),load_cfg={'actor':True},map_location=env.device)
    actor=runner.alg.actor;actor.eval();actor.update_normalization=lambda *a,**k:None
    normalizer={k:v.clone() for k,v in actor.state_dict().items() if k.startswith('obs_normalizer.')}
    for parameter in actor.distribution.parameters():parameter.requires_grad_(False)
    def predict(x):return actor(TensorDict({'actor':x},batch_size=[len(x)]))
    data=np.load(ROOT/'inputs/feedback-data.npz');mask=data['validation']
    x=torch.tensor(data['observation'][~mask],device=env.device)
    y=torch.tensor(data['target'][~mask],device=env.device)
    vx=torch.tensor(data['observation'][mask],device=env.device)
    vy=torch.tensor(data['target'][mask],device=env.device)
    gpu_path=ROOT/'inputs/gpu-feedback-data.pt'
    gpu_audit=json.loads((ROOT/'inputs/gpu-feedback-audit.json').read_text())
    assert hashlib.sha256(gpu_path.read_bytes()).hexdigest()==gpu_audit['dataSha256']
    gpu=torch.load(gpu_path,map_location='cpu',weights_only=False)
    assert gpu['observation'].shape==(1500,100,61) and gpu['target'].shape==(1500,100,14)
    assert bool(gpu['survived'].all()) and int(gpu['validationWorlds'].sum())==25
    train_worlds=~gpu['validationWorlds'];valid_worlds=gpu['validationWorlds']
    gx=gpu['observation'][:,train_worlds].flatten(0,1).to(env.device)
    gy=gpu['target'][:,train_worlds].flatten(0,1).to(env.device)
    gvx=gpu['observation'][:,valid_worlds].flatten(0,1).to(env.device)
    gvy=gpu['target'][:,valid_worlds].flatten(0,1).to(env.device)
    assert all(bool(torch.isfinite(z).all()) for z in (x,y,vx,vy,gx,gy,gvx,gvy))
    del gpu
    optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-4)
    def cpu_screen(policy, destination):
        for yaw in (-.6,0.,.6):
            with (OUT/'cpu-screen.log').open('a') as log:
                subprocess.run([sys.executable,'scripts/evaluate_walk_runtime_cpu.py','--onnx',str(policy),
                    '--physics-profile','kissqy','--output',str(destination/f'cpu-yaw{yaw:g}'),
                    '--speeds','-.2','.2','--seeds','212171','212177','--trials','1',
                    '--seconds','30','--transport-ms','0','--turn-rate',str(yaw)],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'scripts/analyze_walk_runtime.py',str(destination),
            '--seconds','30','--window','17','27'],check=True)
        return json.loads((destination/'cpu-body-tracking.json').read_text())['results']
    baseline_cpu=cpu_screen(ROOT/'inputs/student.onnx',OUT/'cpu-baseline')
    probe_ids=torch.randperm(len(vx),device=env.device)[:8192];vx=vx[probe_ids];vy=vy[probe_ids]
    def validation_rms(xx,yy):
        with torch.no_grad():
            return float(torch.cat([(predict(a)-b).square().flatten() for a,b in zip(xx.split(1024),yy.split(1024))]).mean().sqrt())
    initial=validation_rms(vx,vy);initial_gpu=validation_rms(gvx,gvy)
    previous=0;previous_rms=initial;previous_gpu_rms=initial_gpu;records=[]
    for total in (5,500,1500,3000):
        (OUT/'progress.json').write_text(json.dumps(dict(stage='fit',targetSteps=total)))
        for _ in range(previous,total):
            ids=torch.randint(len(x),(512,),device=env.device)
            aid=torch.randint(len(gx),(512,),device=env.device)
            loss=imitation_loss(x[ids],predict(x[ids]),y[ids])+imitation_loss(gx[aid],predict(gx[aid]),gy[aid])
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        assert all(torch.equal(actor.state_dict()[k],v) for k,v in normalizer.items())
        assert all(bool(torch.isfinite(v).all()) for v in actor.state_dict().values())
        rms=validation_rms(vx,vy);gpu_rms=validation_rms(gvx,gvy)
        stage=OUT/f'step{total}';stage.mkdir(exist_ok=False)
        runner.save(str(stage/'checkpoint.pt'));runner.export_policy_to_onnx(str(stage),'policy.onnx')
        torch.save(dict(optimizer=optimizer.state_dict(),steps=total),stage/'supervised-state.pt')
        session=ort.InferenceSession(str(stage/'policy.onnx'),providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
        probe=vx[:128].cpu().numpy()
        with torch.no_grad():expected=predict(vx[:128]).cpu().numpy()
        actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:z[None]})[0] for z in probe])
        error=float(np.max(abs(actual-expected)));assert error<2e-5
        sha=hashlib.sha256((stage/'policy.onnx').read_bytes()).hexdigest()
        meta=json.loads((ROOT/'inputs/walk.metadata.json').read_text());meta.update(policySha256=sha,hardwareValidated=False)
        meta['walkExecution']['policySha256']=sha
        meta['feedbackDistillation']=dict(steps=total,feedbackUsedAtRuntime=False,accepted=False)
        (stage/'policy.metadata.json').write_text(json.dumps(meta,indent=2))
        audit=dict(stopWeight=stop_weight,steps=total,initialValidationRms=initial,validationRms=rms,initialGpuValidationRms=initial_gpu,gpuValidationRms=gpu_rms,policySha256=sha,
            onnxMaxError=error,normalizerUnchanged=True,feedbackUsedAtRuntime=False,hardwareAccess=False)
        (stage/'audit.json').write_text(json.dumps(audit,indent=2))
        if total==5:
            (OUT/'smoke-audit.json').write_text(json.dumps(audit,indent=2))
            previous=total
            continue
        (OUT/'progress.json').write_text(json.dumps(dict(stage='screen',steps=total)))
        with (stage/'screen.log').open('w') as log:
            subprocess.run([sys.executable,'scripts/server/evaluate_walk_tracking_transfer.py',
                '--candidate',str(stage/'checkpoint.pt'),'--long-hold','--seeds','212151','212157',
                '--output',f'step{total}-screen.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
        screen=json.loads((OUT/f'step{total}-screen.json').read_text())
        lost=sum(len(c['lost']) for c in screen['comparisons'])
        bias={}
        for label in ('baseline','candidate'):
            m=np.concatenate([r['metrics'] for r in screen['results'] if r['policy']==label])
            bias[label]=[float(np.abs(m[:,:,1]-m[:,:,0]).mean()),float(np.abs(m[:,:,3]-m[:,:,2]).mean())]
        keep=lost==0 and rms<=previous_rms*1.05 and gpu_rms<previous_gpu_rms*.9 and all(c<=b+t for c,b,t in zip(bias['candidate'],bias['baseline'],[.01,.03]))
        stop_ok=all(bool(np.all(np.asarray(c['candidate']['stop']) <= np.asarray(c['baseline']['stop'])*1.05+np.array([.001,.01])))
            for comparison in screen['comparisons'] for c in comparison['perCase'] if c['paired'])
        stride_ok=all(c['aggregate']['candidate']['stepDistance']>=.97*c['aggregate']['baseline']['stepDistance']
            for c in screen['comparisons'])
        candidate_cpu=cpu_screen(stage/'policy.onnx',stage/'cpu-screen')
        def cpu_metrics(rows):
            return dict(error=np.mean([r['rolling1sMAE'] for r in rows],0).tolist(),
                stop=float(np.mean([r['stopMeanHorizontalSpeed'] for r in rows])),
                stride=float(np.mean([r['stepDistance'] for r in rows])))
        bb,cc=cpu_metrics(baseline_cpu),cpu_metrics(candidate_cpu)
        cpu_ok=(all(r['completed'] for r in candidate_cpu) and cc['error'][1]<=bb['error'][1]+.01
            and cc['error'][0]<=bb['error'][0]+.003 and cc['stop']<=1.1*bb['stop']+.001
            and cc['stride']>=.95*bb['stride'])
        cpu_directions=all(c['rolling1sMAE'][1]<=b['rolling1sMAE'][1]+.05
            for b,c in zip(baseline_cpu,candidate_cpu))
        assert [r['source'] for r in baseline_cpu]==[r['source'] for r in candidate_cpu]
        gpu_improved=bias['candidate'][1]<.95*bias['baseline'][1]
        keep=keep and gpu_improved and stop_ok and stride_ok and cpu_ok and cpu_directions and all(all(r['survival']) for r in screen['results'])
        records.append(dict(**audit,lost=lost,bias=bias,stopPassed=stop_ok,strideRetained=stride_ok,
            cpuBaseline=bb,cpuCandidate=cc,cpuPassed=cpu_ok,cpuDirectionsPassed=cpu_directions,
            gpuTrackingImproved=gpu_improved,continueTraining=keep,originalScreenAccepted=screen['accepted']))
        (OUT/'stages.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
        if not keep:break
        previous=total;previous_rms=rms;previous_gpu_rms=gpu_rms
    (OUT/'progress.json').write_text(json.dumps(dict(stage='complete',deploymentAccepted=False)))
finally:env.close()
