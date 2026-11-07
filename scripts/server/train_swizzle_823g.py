"""823g body plus original roller mass; transfer the approved roller actor."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence'
POLICY=ROOT/'baseline.onnx'
def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2))
def spec():
 from mjlab_microduck.tasks.swizzle_observer_hd1910 import roller_spec
 from mjlab_microduck.tasks.walk_mass_823g import SERVO_COUNTS, EXTRA_KG
 s=roller_spec();m=s.compile();before=float(m.body_mass.sum())
 for name,count in SERVO_COUNTS.items():
  b=s.body(name);i=m.body(name).id;mass=m.body_mass[i];b.mass=mass+count*.005+EXTRA_KG.get(name,0.)
  b.fullinertia=[float('nan'),0,0,0,0,0];b.iquat=m.body_iquat[i];b.ipos=m.body_ipos[i];b.inertia=m.body_inertia[i]*(b.mass/mass)
 after=float(s.compile().body_mass.sum());assert abs(after-before-.08575682)<1e-8
 save('mass.json',dict(originalKg=before,totalKg=after,addedKg=after-before,bodyReferenceKg=.823,wheelMassPreserved=True))
 return s
def main(mode):
 import mjlab_microduck.tasks.swizzle_observer_hd1910 as task
 original=task.make_cfg
 def cfg(play=False):
  c=original(play);c.scene.entities['robot'].spec_fn=spec;return c
 if mode=='eval':
  import runpy
  task.make_cfg=cfg
  sys.argv=[str(ROOT/'scripts/server/evaluate_swizzle_hd1910.py'),'--policy',str(OUT/'train/policy.onnx'),'--output',str(OUT/'evaluation'),'--envs','16']
  runpy.run_path(sys.argv[0],run_name='__main__');return
 from mjlab.envs import ManagerBasedRlEnv
 from mjlab.rl import RslRlVecEnvWrapper
 from mjlab_microduck.tasks import MicroduckOnPolicyRunner
 from train_walk_823g import warm_start
 import numpy as np, torch, onnxruntime as ort
 from tensordict import TensorDict
 assert hashlib.sha256(POLICY.read_bytes()).hexdigest()=='539e6b2e7e4c7603ddd35ef27bc06742b292f9f3eede567f82111c0975375160'
 c=cfg();c.seed=261106;c.scene.num_envs=64 if mode=='smoke' else 4096
 env=ManagerBasedRlEnv(c,device='cuda:0');rounds=5 if mode=='smoke' else 10000
 try:
  recipe=asdict(task.rl_cfg());r=MicroduckOnPolicyRunner(RslRlVecEnvWrapper(env),recipe,log_dir=str(OUT/mode),device=env.device)
  warm_start(r.alg.actor,str(POLICY),env.device)
  rows=np.random.default_rng(211).normal(0,.35,(128,61)).astype('float32')
  net=ort.InferenceSession(str(POLICY),providers=['CPUExecutionProvider']);expected=np.concatenate([net.run(None,{net.get_inputs()[0].name:row[None]})[0] for row in rows])
  r.alg.actor.eval()
  with torch.no_grad():actual=r.alg.actor(TensorDict({'actor':torch.tensor(rows,device=env.device)},batch_size=[128])).cpu().numpy()
  r.alg.actor.train()
  save(mode+'/parity.json',dict(maxAbsoluteError=float(np.max(np.abs(actual-expected))),atol=2e-5,rtol=1e-5))
  assert np.allclose(actual,expected,atol=2e-5,rtol=1e-5), 'Actor transfer mismatch'

  save(mode+'/contract.json',dict(actorTransfer=True,fullResume=False,freshCritic=True,freshOptimizer=True,syntheticNormalizerCount=1000000,initialActorParityPassed=True,updates=rounds,worlds=c.scene.num_envs,seed=c.seed))
  old=r.save
  def checkpoint(path,*a,**kw):
   old(path,*a,**kw);save('progress.json',dict(stage=mode,updates=env.common_step_counter//recipe['num_steps_per_env'],target=rounds))
  r.save=checkpoint;r.learn(rounds,init_at_random_ep_len=True);r.export_policy_to_onnx(str(OUT/mode),'policy.onnx')
  save(mode+'/complete.json',dict(updates=rounds,elapsedSteps=env.common_step_counter))
 finally:env.close()
if __name__=='__main__':
 mode=sys.argv[1];OUT.mkdir(exist_ok=True)
 if mode=='pipeline':
  try:
   for step in ['smoke','train','eval']:
    with (OUT/(step+'.log')).open('w') as log:subprocess.run([sys.executable,__file__,step],stdout=log,stderr=subprocess.STDOUT,check=True)
   save('progress.json',dict(stage='complete',updates=10000,needsReview=True));(OUT/'job.exit').write_text('0')
  except BaseException:
   (OUT/'job.exit').write_text('1');raise
 else:main(mode)
