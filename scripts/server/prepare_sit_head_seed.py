from pathlib import Path
import json,hashlib,torch,onnx,numpy as np,onnxruntime as ort
from onnx import numpy_helper
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from dataclasses import asdict
ROOT=Path('/home/leiwu30/microduck-training/sit-head-clearance-20261012');BASE=ROOT.parent
p=ROOT/'src/mjlab_microduck/tasks/__init__.py';s=p.read_text();registered='Mjlab-SitHead' in s
s+='''\nfrom .microduck_hd1910_sit_head_env_cfg import make_sit_head_env_cfg,SitHeadRlCfg
for _name,_penalty in [('Control',False),('Repair',True)]:
    register_mjlab_task(task_id='Mjlab-SitHead'+_name+'-Flat-MicroDuck-HD1910',env_cfg=make_sit_head_env_cfg(penalty=_penalty),play_env_cfg=make_sit_head_env_cfg(play=True,penalty=_penalty),rl_cfg=SitHeadRlCfg,runner_cls=MicroduckOnPolicyRunner)
''';p.write_text(s) if not registered else None
model=ROOT/'evidence/current-contract/alpha_sitstand.onnx';tensors={a.name:numpy_helper.to_array(a).copy() for a in onnx.load(model).graph.initializer}
old=next((BASE/'microduck-rl-skills-v3-20261010/logs/rsl_rl/hd1910_sitstand_v1').rglob('model_5000.pt'));state=torch.load(old,map_location='cpu',weights_only=False)
actor=state['actor_state_dict']
for name,value in tensors.items():
 if name.startswith('mlp.') or name=='obs_normalizer._mean':actor[name]=torch.tensor(value)
std=torch.tensor(tensors['onnx::Div_24'])-.01;assert bool((std>=-1e-7).all());std=std.clamp_min(0);actor['obs_normalizer._std']=std;actor['obs_normalizer._var']=std.square();actor['distribution.std_param'].fill_(.05)
state['optimizer_state_dict']['state']={}
for group in state['optimizer_state_dict']['param_groups']:group['lr']=1e-5
state['iter']=0
if 'infos' in state and 'env_state' in state['infos']:state['infos']['env_state']['common_step_counter']=0
seed=ROOT/'logs/rsl_rl/hd1910_sit_head_mechanical/head_seed/seed.pt';seed.parent.mkdir(parents=True);assert not seed.exists();torch.save(state,seed)
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_rl_cfg
cfg={k:v for k,v in asdict(load_rl_cfg('Mjlab-SitHeadControl-Flat-MicroDuck-HD1910').actor).items() if k in ('hidden_dims','activation','obs_normalization','distribution_cfg')}
obs=TensorDict({'actor':torch.tensor(np.random.default_rng(61723).normal(0,.5,(128,61)).astype(np.float32))},batch_size=[128]);m=MLPModel(obs,{'actor':['actor']},'actor',14,**cfg);m.load_state_dict(actor);m.eval()
with torch.no_grad():actual=m(obs).numpy()
session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider']);expected=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in obs['actor'].numpy()]);error=float(np.max(np.abs(actual-expected)));assert error<2e-5,error
(ROOT/'evidence/seed-audit.json').write_text(json.dumps({'teacherSha256':hashlib.sha256(model.read_bytes()).hexdigest(),'seedSha256':hashlib.sha256(seed.read_bytes()).hexdigest(),'deterministicActorOnnxMaxError':error,'criticSource':str(old),'criticSourceSha256':hashlib.sha256(old.read_bytes()).hexdigest(),'criticIsNotDonorOriginal':True,'normalizerCountBorrowedFromCriticCheckpoint':True,'meanAndStdReconstructedFromTeacher':True,'explorationStd':.05,'learningRate':1e-5,'optimizerReset':True,'hardwareAccess':False},indent=2));print({'actorParityError':error})
