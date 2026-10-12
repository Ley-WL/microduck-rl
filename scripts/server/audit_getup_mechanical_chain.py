"""Bounded live action-chain audit with current mechanical physical stops."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.getup_mechanical_contract import encode_target,NAMES

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
cfg=load_env_cfg('Mjlab-GetupMechanical-Flat-MicroDuck-HD1910');cfg.scene.num_envs=64
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    obs,_=env.reset(seed=61217)
    assert obs['actor'].shape==(64,61)
    term=env.action_manager.get_term('joint_pos');assert tuple(term._target_names)==NAMES
    filtered=np.zeros((64,14));errors=[];history=[];resets=0;discrepancies=[]
    for step in range(40):
        action=np.random.default_rng(61700+step).normal(0,1.4,(64,14)).astype(np.float32)
        filtered=.45*filtered+.55*action.astype(np.float64)
        expected=encode_target(filtered+term._home64.detach().cpu().numpy())
        obs,reward,terminated,truncated,_=env.step(torch.tensor(action,device=env.device))
        done=(terminated|truncated).cpu().numpy()
        actual=term.position_target.detach().cpu().numpy()
        errors.append(float(np.max(np.abs(expected-actual))))
        if errors[-1]>2e-6:
            world,joint=np.unravel_index(np.argmax(np.abs(expected-actual)),expected.shape)
            discrepancies.append(dict(step=step,world=int(world),joint=NAMES[joint],expected=float(expected[world,joint]),actual=float(actual[world,joint]),filteredExpected=float(filtered[world,joint]),filteredActual=float(term._filtered_actions[world,joint])))
        wanted=action.copy();wanted[done]=0
        history.append(float(np.max(np.abs(obs['actor'][:,34:48].cpu().numpy()-wanted))))
        filtered[done]=0;resets+=int(done.sum())
    args.output.write_text(json.dumps(dict(worlds=64,steps=40,resets=resets,
        targetContractMaxAbsErrorRad=max(errors),rawHistoryMaxAbsError=max(history),
        discrepancies=discrepancies,
        hardwareAccess=False,motionAcceptance=False,contract='EMA(raw) -> HOME -> mechanical clip -> current encoder')))
    assert max(errors)<2e-6 and max(history)<2e-6
finally:env.close()
