"""BAM pose sweep before training; physical contact validation, not policy success."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from bam.mujoco import MujocoController
from evaluate_skills_cpu import build_model, quaternion


def check(metadata, trials=16, knees=(0.,1.05,1.15,1.2,1.25,1.35), hip=.4079, seconds=3.):
    model, bam, ground = build_model(metadata, contact_repair=True)
    home = np.array(metadata['homeRadians'])
    rng = np.random.default_rng(5109)
    groups = []
    for knee in knees:
        target = home.copy()
        if knee:
            for index,value in metadata['sitJointOverrides'].items():
                target[int(index)] = value
            target[3],target[12] = knee,-knee
            target[2],target[11] = -hip,hip
        samples = []
        for _ in range(trials):
            data = mujoco.MjData(model)
            controller = MujocoController(bam, metadata['jointOrder'],model,data,vin_drop_gain=0.,vin_min=4.)
            data.qpos[:7] = [0,0,.12,*quaternion(*rng.uniform(-.04,.04,2),0.)]
            data.qpos[controller.qpos_indexes] = target+rng.uniform(-.025,.025,14)
            mujoco.mj_forward(model,data)
            data.qpos[2] += .003-ground.lowest_mesh_z(model,data)
            mujoco.mj_forward(model,data)
            controller.q_target[:] = target
            bam.actuator.q_target_smooth = data.qpos[controller.qpos_indexes].copy()
            rows = []
            for tick in range(round(seconds/model.opt.timestep)):
                controller.update()
                mujoco.mj_step(model,data)
                if tick >= round((seconds-1.)/model.opt.timestep):
                    mujoco.mj_kinematics(model,data)
                    trunk = model.body('trunk_base').id
                    tilt = np.degrees(np.arccos(np.clip(data.xmat[trunk].reshape(3,3)[2,2],-1.,1.)))
                    error = np.abs(target-data.qpos[controller.qpos_indexes]).max()
                    rows.append([tilt,data.xpos[trunk,2],error])
            samples.append(np.mean(rows,axis=0))
        values=np.array(samples)
        groups.append({'kneeTargetRadians':knee,'hipTargetRadians':hip,'holdSeconds':seconds,'trials':trials,
                       'tiltMaxDeg':float(values[:,0].max()),
                       'heightMeanM':float(values[:,1].mean()),
                       'maxJointErrorMeanRad':float(values[:,2].mean()),
                       'uprightTrials':int((values[:,0]<20.).sum())})
    return {'hardwareValidated':False,'policyEvaluation':False,'groups':groups}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--knees',type=float,nargs='+',default=[0.,1.05,1.15,1.2,1.25,1.35])
    p.add_argument('--hip',type=float,default=.4079)
    p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--trials',type=int,default=16)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=check(json.loads(args.metadata.read_text()),args.trials,args.knees,args.hip,args.seconds)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
