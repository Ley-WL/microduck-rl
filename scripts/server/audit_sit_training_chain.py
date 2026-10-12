"""Bounded live-simulator audit of gain DR and the runtime action contract."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from bam.mjlab import BamActuator


def main(args):
    usage=int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert usage<200,'GPU6 occupied; do not interfere'
    cfg=load_env_cfg(args.task)
    cfg.scene.num_envs=64
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.onnx),sess_options=options,providers=['CPUExecutionProvider'])
    try:
        obs,_=env.reset(seed=9017)
        actuators=[a for a in env.scene['robot'].actuators if isinstance(a,BamActuator)]
        assert len(actuators)==1
        actuator=actuators[0];gains=actuator.kp_scale.clone();damping=actuator.kd_scale.clone()
        assert bool(((gains>=.8)&(gains<=1.2)).all())
        assert bool(((damping>=.9)&(damping<=1.1)).all())
        term=env.action_manager.get_term('joint_pos');home=env.scene['robot'].data.default_joint_pos
        lower=torch.tensor([cfg.actions['joint_pos'].clip[n][0] for n in term._target_names],device=env.device)
        upper=torch.tensor([cfg.actions['joint_pos'].clip[n][1] for n in term._target_names],device=env.device)
        filtered=torch.zeros_like(home);errors=[];history_errors=[];gain_errors=[]
        for _ in range(80):
            values=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0]
                                   for row in obs['actor'].cpu().numpy()])
            action=torch.from_numpy(values).to(env.device)
            filtered=.45*filtered+.55*action
            expected=(home+filtered).clamp(lower,upper)
            obs,reward,terminated,truncated,_=env.step(action)
            assert not bool((terminated|truncated).any()),'Audit reset; cannot compare state chains'
            errors.append(float((expected-term.position_target).abs().max()))
            history_errors.append(float((obs['actor'][:,34:48]-action).abs().max()))
            gain_errors.append(float((actuator.kp_scale-gains).abs().max()))
        assert max(errors)<1e-6 and max(history_errors)<1e-6 and max(gain_errors)==0.
        # Effective BAM proportional factor after compute, not only cfg values.
        torch.testing.assert_close(actuator._bam_model.actuator.kp,actuator._base_kp*gains,atol=0,rtol=0)
        report={'hardwareAccess':False,'motionAcceptance':False,'task':args.task,'worlds':64,'steps':80,
            'kpScale':{'min':float(gains.min()),'max':float(gains.max()),'std':float(gains.std()),'values':gains.cpu().flatten().tolist()},
            'kdScale':{'min':float(damping.min()),'max':float(damping.max())},
            'baseKp':actuator._base_kp,'maxGainChangeAcrossCompute':max(gain_errors),
            'maxAbsoluteTargetContractErrorRad':max(errors),'maxRawHistoryContractErrorRad':max(history_errors),
            'scope':'One seed, active simulator. Does not identify physical HD1910 firmware P/D or prove complete CPU/GPU dynamics parity.'}
        args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',default='Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910')
    main(p.parse_args())
