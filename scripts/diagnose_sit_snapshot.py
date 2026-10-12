"""Closed-loop sensitivity from a captured pose; hidden actuator state is unknown."""
import importlib.util,json
from collections import deque
from pathlib import Path
import mujoco,numpy as np,onnxruntime as ort
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('skill_cpu',root/'scripts/evaluate_skills_cpu.py')
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
directory=root.parent/'microduck-observer/docs/实测记录/附件/平台/2026-10-10/sit-trace'
capture=json.loads((directory/'session.json').read_text(encoding='utf-8'))
frames=[r for r in capture['frames'] if r['type']=='inference']
sit=next(r for r in frames if r['kind']=='sitstand_sit')
source=[r for r in frames if r['mono']<sit['mono']][-1]
meta=json.loads((root.parent/'microduck-observer/models/alpha_sitstand.metadata.json').read_text())
home=np.array(meta['homeRadians']);names=meta['jointOrder'];obs0=np.array(sit['observation'])
angles=home+obs0[6:20];gravity=obs0[3:6]
roll,pitch=np.arctan2(-gravity[1],-gravity[2]),np.arcsin(np.clip(gravity[0],-1,1))
opt=ort.SessionOptions();opt.intra_op_num_threads=1
session=ort.InferenceSession(str(root.parent/'microduck-observer/models/alpha_sitstand.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
results=[]
for delay in (3,5,7):
    for mode in ('reset','source-history','pose-ema'):
        model,bam,ground=cpu.build_model(meta)
        data=mujoco.MjData(model)
        controller=cpu.MujocoController(bam,names,model,data,vin_drop_gain=0.,vin_min=4.)
        qadr,vadr=controller.qpos_indexes,controller.dof_indexes
        data.qpos[:7]=[0,0,.12,*cpu.quaternion(roll,pitch,0)]
        data.qpos[qadr]=angles
        data.qvel[vadr]=obs0[20:34];data.qvel[3:6]=obs0[:3]
        mujoco.mj_forward(model,data)
        data.qpos[2]+=.003-ground.lowest_mesh_z(model,data)
        mujoco.mj_forward(model,data)
        controller.q_target[:]=angles;bam.actuator.q_target_smooth=angles.copy()
        history=deque([angles.copy() for _ in range(delay+1)],maxlen=delay+1)
        previous=np.zeros(14);filtered=np.zeros(14)
        if mode=='source-history':
            previous=np.array(source['rawActionRadians'])+np.array(source['homeRadians'])-home
            filtered=np.array(source['filteredActionRadians'])+np.array(source['homeRadians'])-home
        if mode=='pose-ema':filtered=angles-home
        velocity=obs0[20:34].copy();gyro=obs0[:3].copy()
        target_frames=[];heights=[];tilts=[]
        trunk=model.body('trunk_base').id
        for tick in range(300):
            sit_command=tick<200
            measured_gyro,g=cpu.root_body_imu_observation(model,data,trunk)
            velocity=.16*velocity+.84*data.qvel[vadr]
            gyro=.25*gyro+.75*measured_gyro
            commands=np.zeros(13);commands[0]=float(sit_command)
            observation=np.concatenate([gyro,g,data.qpos[qadr]-home,velocity,previous,commands]).astype(np.float32)
            action=session.run(None,{session.get_inputs()[0].name:observation[None]})[0][0]
            filtered=.45*filtered+.55*action;previous=action.copy()
            bounds=model.jnt_range[[model.joint(n).id for n in names]]
            target=np.clip(home+filtered,bounds[:,0],bounds[:,1]);target_frames.append(target.copy())
            for _ in range(4):
                history.append(target.copy());controller.q_target[:]=history[0];controller.update();mujoco.mj_step(model,data)
            heights.append(float(data.xpos[trunk,2]));tilts.append(float(np.degrees(np.arccos(np.clip(data.xmat[trunk].reshape(3,3)[2,2],-1,1)))))
        targets=np.degrees(np.array(target_frames[:100]))
        results.append({'mode':mode,'queuedDelayMs':delay*5,'kneeVariationFirst2sDeg':np.abs(np.diff(targets[:,[3,12]],axis=0)).sum(axis=0).tolist(),
            'kneeRangeFirst2sDeg':np.stack([targets[:,[3,12]].min(axis=0),targets[:,[3,12]].max(axis=0)],axis=1).tolist(),
            'sitHeightLast1sM':float(np.mean(heights[150:200])),'standHeightLast1sM':float(np.mean(heights[250:300])),
            'maxTiltDeg':max(tilts),'finite':bool(np.isfinite(data.qpos).all())})
report={'hardwareValidated':False,'limitations':'Captured pose and filtered velocity approximate initial physical state; unobserved height, contacts and actuator hidden states reconstructed. Simulation kp5 is not a verified map of real firmware P6. Does not isolate hardware cause.', 'results':results}
output=root/'artifacts/sit-transition-diagnostic-20261010/snapshot-start.json'
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
