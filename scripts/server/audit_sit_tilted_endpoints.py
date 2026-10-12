"""Score saved paired poses with unchanged task rewards; no dynamics/hardware.

Velocities are set to zero for a static counterfactual. This is the subtotal
of listed geometric task terms, not full rollout return or a PPO advantage.
"""
import argparse,json,re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks.microduck_hd1910_sit_contact_env_cfg import get_sit_contact_spec
from mjlab_microduck.tasks import mdp

TERMS=('posture_pose_legs','posture_pose_l1','head_pose_tracking','posture_height',
       'posture_height_sharp','posture_height_l1','upright_linear','posture_composite','posture_stillness')


def score(qpos,flag,cfg,home):
    qpos=torch.tensor(qpos,dtype=torch.float32)
    n=len(qpos);joint=qpos[:,7:]
    assert joint.shape[1]==14
    asset=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=qpos[:,:3],
        root_link_quat_w=qpos[:,3:7],joint_pos=joint,default_joint_pos=home.repeat(n,1),
        root_link_lin_vel_w=torch.zeros(n,3)))
    class Scene(dict):pass
    scene=Scene(robot=asset);scene.terrain=SimpleNamespace(env_origins=torch.zeros(n,3))
    command=SimpleNamespace(alpha=torch.full((n,),float(flag)),command=torch.tensor([[flag,0.,0.]]).repeat(n,1))
    env=SimpleNamespace(scene=scene,device='cpu',_head_pose_neck_ids=torch.tensor([5,6,7,8]),
        _head_pose_bl_ids=torch.zeros(4,dtype=torch.long),_head_pose_bl_mask=torch.zeros(4),
        command_manager=SimpleNamespace(get_term=lambda _:command,
          get_command=lambda name:torch.tensor([[flag,0.,0.]]).repeat(n,1) if name=='twist' else torch.zeros(n,4)))
    values={}
    with patch.object(mdp,'_servo_joint_pos',return_value=joint),patch.object(
            mdp,'_servo_default_joint_pos',return_value=home.repeat(n,1)):
        for name in TERMS:
            term=cfg.rewards[name]
            values[name]=(term.func(env,**term.params)*term.weight).detach().numpy()
    return {name:float(value.mean()) for name,value in values.items()},float(sum(values.values()).mean())


def main(args):
    cfg=load_env_cfg(args.task)
    model=get_sit_contact_spec().compile()
    home=torch.tensor([next(v for k,v in HOME_FRAME.joint_pos.items()
        if re.fullmatch(k,model.joint(j).name)) for j in range(1,model.njnt)])
    assert model.nq==21
    results=[]
    for folder in args.folders:
        for trial in ('stand_cycle-9001-0','sit_cycle-9001-0'):
            path=folder/(trial+'.npz')
            trace=json.loads(path.with_suffix('.json').read_text())
            poses=np.load(path)['qpos'];assert len(poses)==len(trace['trace'])
            windows=[]
            for start,end in ((5.,8.),(11.,14.),(17.,20.)):
                ids=[i for i,r in enumerate(trace['trace']) if start<=r['time']<end]
                assert len(ids)>=100
                flag=float(trace['trace'][ids[0]]['sit'])
                assert all(float(trace['trace'][i]['sit'])==flag for i in ids)
                terms,subtotal=score(poses[ids],flag,cfg,home)
                windows.append({'start':start,'end':end,'commandSit':flag,
                    'meanTiltDeg':float(np.mean([trace['trace'][i]['tiltDeg'] for i in ids])),
                    'actualStableFraction':float(np.mean([trace['trace'][i]['stable'] for i in ids])),
                    'staticTaskSubtotal':subtotal,'terms':terms})
            results.append({'folder':str(folder),'trial':trial,'trialSuccess':trace['result']['success'],'windows':windows})
    # Independent ideal endpoints make both command directions auditable.
    ideal=[]
    for posture in (0.,1.):
        q=np.zeros(21);q[3]=1.;q[7:]=home.numpy()
        if posture:
            for j,v in cfg.rewards['posture_pose_legs'].params['sit_overrides'].items():q[7+j]=v
        q[2]=cfg.commands['twist'].sit_z if posture else cfg.commands['twist'].stand_z
        for flag in (0.,1.):
            terms,total=score(q[None],flag,cfg,home)
            ideal.append({'postureSit':posture,'commandSit':flag,'staticTaskSubtotal':total,'terms':terms})
    output={'hardwareAccess':False,'diagnosticOnly':True,'task':args.task,
        'scope':'Listed 9 geometric task terms, alpha=command at settled endpoint, velocities counterfactually zero. Excludes torque/contact/action/temporal and rollout history costs, not total return or proof of a unique cause.',
        'ideal':ideal,'pairedSavedPoses':results}
    args.output.write_text(json.dumps(output,indent=2));print(json.dumps({'ideal':ideal,'results':results},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folders',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',default='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    main(p.parse_args())
