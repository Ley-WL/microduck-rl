"""Evaluate 30 commands using the common HD1910 runtime approximation.

WALK_POLICY_PATH selects an ONNX; WALK_OUTPUT_DIR selects output.
Default: the accepted observer5000 policy. Legacy EMA requires label initial.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MICRODUCK_WALK_SURFACE_INPUTS', str(ROOT/'models/walk_observer_hd1910_5000'))
import sys,json,hashlib
from dataclasses import dataclass
import numpy as np
import torch
import onnxruntime as ort
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import walk_runtime_eval as base

LABEL=sys.argv[1] if len(sys.argv)>1 else "observer5000"
assert LABEL in ('initial','original5000','repair100','observer5000')
OUT=Path(os.environ.get('WALK_OUTPUT_DIR',str(ROOT/'evidence/input-response'/LABEL)))
OUT.mkdir(parents=True,exist_ok=True)
POLICIES={LABEL:ROOT/'models/walk_observer_hd1910_5000/policy.onnx'}
CASES = []
for speed in (.2,.3,.4):
    for sign,name in [(1,'forward'),(-1,'back')]:
        CASES.append((f'{name}{round(speed*100)}',(sign*speed,0,0)))
for speed in (.2,.3,.4,.5,.75,1.):
    for sign,name in [(1,'left'),(-1,'right')]:
        CASES.append((f'yaw_{name}_{speed:g}',(0,0,sign*speed)))
for speed in (.2,.3,.4):
    for yaw in (.5,1.):
        for sign,name in [(1,'left'),(-1,'right')]:
            CASES.append((f'arc_{speed:g}_{name}_{yaw:g}',(speed,0,sign*yaw)))

class CompareAction(base.RuntimeWalkAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if LABEL!='initial': self._alpha.fill_(1.)
    def process_actions(self,actions):
        # Verified metadata: scale=1 for all; old head/legs EMA .5/.7, new no EMA.
        self._filtered=torch.where(self._initialized,(1-self._alpha)*self._filtered+self._alpha*actions,actions)
        self._initialized.fill_(True);self._raw_actions.copy_(actions)
        bounded=(self._home+self._filtered.double()).clamp(self._bounds[:,0],self._bounds[:,1])
        encoded=torch.floor(self._refs+bounded/base.ENCODER_STEP*self._directions+.5)
        self._processed_actions=((encoded-self._refs)*base.ENCODER_STEP*self._directions).float()

@dataclass(kw_only=True)
class CompareActionCfg(base.RuntimeWalkActionCfg):
    def build(self,env): return CompareAction(self,env)

def gyro(env,**kwargs): return env.scene['robot'].data.root_link_ang_vel_b

def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

POLICIES[LABEL]=Path(os.environ.get('WALK_POLICY_PATH',str(POLICIES[LABEL])))
opts=ort.SessionOptions();opts.intra_op_num_threads=1
session=ort.InferenceSession(str(POLICIES[LABEL]),sess_options=opts,providers=['CPUExecutionProvider'])
cfg=base.make_cfg(play=True);cfg.seed=191057;cfg.scene.num_envs=16;cfg.episode_length_s=15.
cfg.actions['joint_pos']=CompareActionCfg(**vars(cfg.actions['joint_pos']))
cfg.observations['actor'].enable_corruption=False
cfg.observations['actor'].terms['base_ang_vel'].func=gyro
cfg.observations['actor'].terms['base_ang_vel'].params={}
torch.manual_seed(191057)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
robot=env.scene['robot']
hips,_=robot.find_joints(['left_hip_yaw','right_hip_yaw'],preserve_order=True)
passive,_=robot.find_joints(['passive_left_hip_yaw_backlash','passive_right_hip_yaw_backlash'],preserve_order=True)
results={};traces={};initial_hashes={}
save('contract.json',dict(label=LABEL,policySha256=hashlib.sha256(POLICIES[LABEL].read_bytes()).hexdigest(),
     worlds=16,seed=191057,dt=.02,sequence='0-2s idle, 2-8s input, 8-10s stop',
     measurement='body-frame vx/vy/yaw; mean 4-8s; failed worlds excluded after first failure',
     physics='Common replica backlash, HD1910 BAM P5, 7.4V, 14ms actuator delay, mechanical limits',
     runtime='scale1; metadata EMA; encoder rounding; previous feedback velocity; command smoothing .2',
     caveat='Approximate runtime simulation, ideal body gyro; not real measured response or exact board calibration replay'))
try:
    for case,command in CASES:
        torch.manual_seed(191057);obs,_=env.reset(seed=191057)
        initial_hashes[case]=hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes()+env.sim.data.qvel.cpu().numpy().tobytes()).hexdigest()
        failed=np.zeros(16,bool);vel=[];hip=[];fail=[]
        with torch.no_grad():
            for tick in range(500):
                term=env.command_manager.get_term('twist')
                cmd=command if 100<=tick<400 else (0,0,0)
                term.vel_command_b[:]=torch.tensor(cmd,device=env.device)
                term.vel_command_w[:]=term.vel_command_b
                term.is_standing_env[:]=False;term.is_world_env[:]=False;term.is_heading_env[:]=False;term.time_left[:]=100.
                rows=obs['actor'].cpu().numpy()
                action=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in rows])
                assert action.shape==(16,14) and np.isfinite(action).all()
                obs,_,terminated,truncated,_=env.step(torch.tensor(action,device=env.device))
                failed|=(terminated|truncated).cpu().numpy()
                value=torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),-1).cpu().numpy()
                q=(robot.data.joint_pos[:,hips]+robot.data.joint_pos[:,passive]).cpu().numpy()
                value[failed]=np.nan;q[failed]=np.nan
                vel.append(value);hip.append(q);fail.append(failed.copy())
        v=np.asarray(vel);h=np.asarray(hip);f=np.asarray(fail)
        mean=np.nanmean(v[200:400],axis=(0,1));std=np.nanstd(v[200:400],axis=(0,1))
        response=[];reached=[]
        for axis,target in enumerate(command):
            if not target: response.append(None);reached.append(None);continue
            times=[]
            for world in range(16):
                good=(v[100:400,world,axis]*np.sign(target)>=.7*abs(target)) & ~f[100:400,world]
                hit=np.flatnonzero(np.convolve(good.astype(int),np.ones(25,dtype=int),mode='valid')==25)
                if len(hit):times.append(float(hit[0]*.02))
            response.append(float(np.median(times)) if times else None);reached.append(len(times))
        stop=[]
        for world in range(16):
            good=(np.linalg.norm(v[400:,world,:2],axis=-1)<.03)&(abs(v[400:,world,2])<.1)&~f[400:,world]
            hit=np.flatnonzero(np.convolve(good.astype(int),np.ones(25,dtype=int),mode='valid')==25)
            if len(hit):stop.append(float(hit[0]*.02))
        survivors=~f[-1]
        result=dict(command=command,survived=int(survivors.sum()),worlds=16,
            meanVelocity=mean.tolist(),velocityStd=std.tolist(),response70Seconds=response,responseReached=reached,
            stopSecondsMedian=float(np.median(stop)) if stop else None,stopReached=len(stop),
            hipExcursionDeg=np.degrees(np.median(np.ptp(h[200:400,survivors],axis=0),axis=0)).tolist() if survivors.any() else None,
            hipMeanDeg=np.degrees(np.nanmean(h[200:400],axis=(0,1))).tolist(),
            validSamples=int(np.isfinite(v[200:400]).all(-1).sum()))
        results[case]=result;traces[case+'_velocity']=v;traces[case+'_hip']=h;traces[case+'_failed']=f
        save('results.json',results);save('initial-hashes.json',initial_hashes)
        print(LABEL,case,result['survived'],mean.tolist(),flush=True)
finally:env.close()
np.savez_compressed(OUT/'traces.npz',**traces)
save('status.json',dict(stage='complete',cases=len(results)))
