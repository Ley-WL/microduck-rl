"""Small paired current-contract sitstand experiment. No hardware or deployment."""
import os,time,json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/fixedlr-control';PY=ROOT.parent/'microduck-rl/.venv/bin/python'
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='offline',MICRODUCK_WARM_START='1')
sys.path.insert(0,str(ROOT/'src'))
SEED=62437

def status(stage,**fields):
 (OUT/'progress.json').write_text(json.dumps(dict(stage=stage,**fields),indent=2))

def capacity(stage):
 available=0
 while available<2:
  line=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();free,util=map(int,line.split(','))
  available=available+1 if free>12288 and util<30 else 0
  status('capacity-check',nextStage=stage,gpu0FreeMiB=free,gpu0Utilization=util)
  if available<2:time.sleep(5)

def run(args,path):
 with path.open('w') as f:subprocess.run(list(map(str,args)),cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
 assert (OUT/'config-tests.exit').read_text().strip()=='0'
 original=json.loads((ROOT/'evidence/current-contract/alpha_sitstand.metadata.json').read_text())
 for iterations,worlds,stage in [(5,64,'smoke'),(20,1024,'main')]:
  for group in ['Control','Repair']:
   task='Mjlab-SitPitchFixed'+group+'-Flat-MicroDuck-HD1910';name=group.lower()+'-'+stage+'-62437'
   logs=ROOT/'logs/rsl_rl/hd1910_sit_pitch_fixedlr';existing=list(logs.glob('*'+name));reuse=bool(existing)
   assert not reuse or (stage=='smoke' and group=='Control' and len(existing)==1 and (existing[0]/('model_'+str(iterations-1)+'.pt')).exists()),'Inspect existing run'
   if not reuse:capacity(name)
   status('training',group=group,stageName=stage,iterations=iterations,worlds=worlds,trainingSeed=SEED)
   if not reuse:run([PY.parent/'train',task,'--env.scene.num-envs',worlds,'--gpu-ids','all','--agent.max-iterations',iterations,'--agent.seed',SEED,'--agent.resume','True','--agent.load-run','pitch_seed','--agent.load-checkpoint','seed.pt','--agent.run-name',name,'--agent.logger','tensorboard'],OUT/(name+'.log'))
   checkpoints=list(logs.glob('*'+name+'/model_'+str(iterations-1)+'.pt'));assert len(checkpoints)==1
   import torch
   state=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
   assert all(bool(torch.isfinite(v).all()) for label in ['actor_state_dict','critic_state_dict'] for v in state[label].values())
   steps={int(v['step']) for v in state['optimizer_state_dict']['state'].values() if 'step' in v};assert steps=={iterations*20},steps
   assert all(abs(g['lr']-1e-5)<1e-12 for g in state['optimizer_state_dict']['param_groups'])
   capacity('export-'+name);policy=OUT/(name+'.onnx')
   run([PY,'scripts/export.py',task,'--checkpoint-file',checkpoints[0],'--onnx-file',policy,'--num-envs',1,'--device','cuda:0'],OUT/(name+'-export.log'))
   import onnxruntime as ort,numpy as np
   from tensordict import TensorDict
   from rsl_rl.models import MLPModel
   from mjlab.tasks.registry import load_rl_cfg
   from dataclasses import asdict
   import mjlab_microduck.tasks
   cfg={k:v for k,v in asdict(load_rl_cfg(task).actor).items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
   obs=TensorDict({'actor':torch.tensor(np.random.default_rng(62437).normal(0,.5,(128,61)).astype(np.float32))},batch_size=[128]);model=MLPModel(obs,{'actor':['actor']},'actor',14,**cfg);model.load_state_dict(state['actor_state_dict']);model.eval()
   with torch.no_grad():expected=model(obs).numpy()
   session=ort.InferenceSession(str(policy),providers=['CPUExecutionProvider']);actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in obs['actor'].numpy()]);error=float(np.abs(actual-expected).max());assert error<2e-5,error
   meta=dict(original,policySha256=hashlib.sha256(policy.read_bytes()).hexdigest(),task=task,artifactType='training-policy',diagnosticOnly=True,trainingIterations=iterations,standingHeadGroundWeight=-1,standingForwardPitchWeight=0 if group=='Control' else -2)
   policy.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2));(OUT/(name+'-audit.json')).write_text(json.dumps({'finite':True,'optimizerStep':sorted(steps),'normalizerIncluded':True,'actorOnnxMaxError':error,'hardwareAccess':False},indent=2))
 for group in ['control','repair']:
  for condition in ['normal','stress']:
   name=group+'-'+condition;status('cpu-screen',name=name)
   cpu=dict(ENV,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1')
   args=[str(PY),'scripts/evaluate_sit_head_cpu.py','--onnx',str(OUT/(group+'-main-62437.onnx')),'--profile','runtime','--native-sensors','--walk-onnx',str(ROOT/'evidence/current-contract/xgoduck_walk.onnx'),'--stand-handoff-seconds','1','--seeds','817031','929017','--trials','2','--scenario','sit_cycle','stand_cycle','--output',str(OUT/name)]
   if condition=='stress':args+=['--transition-stress']
   with (OUT/(name+'.log')).open('w') as f:subprocess.run(args,cwd=ROOT,env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
 status('complete',hardwareAccess=False,accepted=False)
if __name__=='__main__':
 try:main();(OUT/'job.exit').write_text('0')
 except BaseException:(OUT/'job.exit').write_text('1');raise
