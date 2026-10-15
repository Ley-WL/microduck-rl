"""Bounded refinement: keep actor normalizer, reset Adam, reduce exploration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--iterations',type=int,default=400)
p.add_argument('--envs',type=int,default=1024)
p.add_argument('--task',default='Mjlab-Dance10Stable-Flat-MicroDuck-HD1910')
p.add_argument('--exploration',type=float,default=.10)
p.add_argument('--seed',type=int,default=81423)
a=p.parse_args()
torch.manual_seed(a.seed)
task=a.task
cfg=load_env_cfg(task);cfg.scene.num_envs=a.envs
env=ManagerBasedRlEnv(cfg,device='cuda:0')
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
try:
    runner=load_runner_cls(task)(RslRlVecEnvWrapper(env),asdict(load_rl_cfg(task)),log_dir=str(out),device=env.device)
    runner.load(a.checkpoint,load_cfg={'actor':True,'critic':True},map_location=env.device)
    env.common_step_counter=0
    assert not runner.alg.optimizer.state
    actor=runner.alg.actor
    normalizer={k:v.clone() for k,v in actor.obs_normalizer.state_dict().items()}
    actor.update_normalization=lambda obs: None
    distribution=actor.distribution
    assert distribution.std_type=='scalar', distribution.std_type
    with torch.no_grad(): distribution.std_param.fill_(a.exploration)
    runner.learn(a.iterations,init_at_random_ep_len=True)
    assert all(torch.equal(v,actor.obs_normalizer.state_dict()[k]) for k,v in normalizer.items())
    assert all(torch.isfinite(v).all() for v in actor.state_dict().values())
    runner.save(str(out/'final.pt'))
    (out/'audit.json').write_text(json.dumps(dict(iterations=a.iterations,worlds=a.envs,
        actorNormalizerUnchanged=True,seed=a.seed,initialExplorationStd=a.exploration,
        loadedActorCritic=True,loadedAdam=False,learningRate=load_rl_cfg(task).algorithm.learning_rate,
        finite=True,accepted=False),indent=2))
finally: env.close()
