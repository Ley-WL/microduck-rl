"""Matched frozen-load force parity + actual assembled CPU/GPU model audit.

No skill acceptance or full CPU/Warp rollout equivalence claim. No hardware.
"""
import argparse,hashlib,importlib.util,json,subprocess
from pathlib import Path
import numpy as np
import torch
import mujoco
from bam.model import load_model
from bam.actuator import TorchBackend
from bam.mjlab import BamActuator
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('cpu_skill',root/'scripts/evaluate_skills_cpu.py')
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)

def main(a):
    assert int(subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<200,'GPU6 occupied'
    assert not a.output.exists(),'Inspect existing audit'
    metadata=json.loads(a.onnx.with_suffix('.metadata.json').read_text())
    cm,bam,ground=cpu.build_model(metadata)
    cfg=load_env_cfg('Mjlab-SitStandProcessWeakProcess-Flat-MicroDuck-HD1910',play=True);cfg.scene.num_envs=4
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env.reset(seed=9031);gm=env.sim.mj_model
        actuator=next(x for x in env.scene['robot'].actuators if isinstance(x,BamActuator))
        mismatches=[];checks=0
        for i in range(cm.njnt):
            name=cm.joint(i).name
            if not name or name.startswith('passive_'):continue
            try:j=gm.joint('robot/'+name).id
            except KeyError:continue
            for field in ('jnt_range','jnt_axis','jnt_pos'):
                checks+=1
                if not np.allclose(getattr(cm,field)[i],getattr(gm,field)[j],atol=1e-7,rtol=1e-6):
                    mismatches.append({'kind':'joint','name':name,'field':field,'cpu':getattr(cm,field)[i].tolist(),'gpu':getattr(gm,field)[j].tolist()})
        for i in range(1,cm.nbody):
            name=cm.body(i).name
            if not name:continue
            try:j=gm.body('robot/'+name).id
            except KeyError:continue
            for field in ('body_mass','body_inertia','body_ipos','body_iquat'):
                checks+=1
                if not np.allclose(getattr(cm,field)[i],getattr(gm,field)[j],atol=1e-7,rtol=1e-6):
                    mismatches.append({'kind':'body','name':name,'field':field,'cpu':np.asarray(getattr(cm,field)[i]).tolist(),'gpu':np.asarray(getattr(gm,field)[j]).tolist()})
        for i in range(cm.ngeom):
            name=cm.geom(i).name
            if not name:continue
            try:j=gm.geom('robot/'+name).id
            except KeyError:continue
            for field in ('geom_contype','geom_conaffinity','geom_condim','geom_friction','geom_solref','geom_solimp','geom_priority'):
                checks+=1
                if not np.allclose(getattr(cm,field)[i],getattr(gm,field)[j],atol=1e-7,rtol=1e-6):
                    mismatches.append({'kind':'geom','name':name,'field':field,'cpu':np.asarray(getattr(cm,field)[i]).tolist(),'gpu':np.asarray(getattr(gm,field)[j]).tolist()})
        # Snapshots come from real CPU integration, including actual constraint loads.
        samples=[];old_diffs=[];new_diffs=[];torque_diffs=[];dt=float(cm.opt.timestep)
        for sit in (False,True):
            d=mujoco.MjData(cm);home=np.asarray(metadata['homeRadians']);pose=home.copy()
            if sit:
                for index,value in metadata['sitJointOverrides'].items():pose[int(index)]=value
            d.qpos[:7]=[0,0,.15,1,0,0,0];d.qpos[7:]=pose
            mujoco.mj_forward(cm,d);d.qpos[2]+=.003-ground.lowest_mesh_z(cm,d);mujoco.mj_forward(cm,d)
            controller=cpu.MujocoController(bam,metadata['jointOrder'],cm,d,vin_drop_gain=0.,vin_min=4.)
            controller.q_target[:]=pose;bam.actuator.kp=4.;bam.actuator.q_target_smooth=pose.copy()
            tm=load_model(str(root/'src/mjlab_microduck/robot/hd1910/1910_m6.json'))
            tm.actuator.backend=TorchBackend();tm.actuator.vin=7.4;tm.actuator.kp=4.;tm.actuator.max_current=None
            tm.actuator.q_target_smooth=torch.tensor(pose[None],dtype=torch.float32,device=env.device)
            actuator.friction_scale.fill_(1.)
            for step in range(240):
                ids=controller.dof_indexes;q=d.qpos[controller.qpos_indexes].copy();v=d.qvel[ids].copy()
                f=cpu._controller_module.dof_friction_force(d,ids)
                wrong=cpu._controller_module.dof_friction_force(d,controller.joint_indexes)
                ext=-d.qfrc_bias[ids]+d.qfrc_constraint[ids]-f
                bad_ext=-d.qfrc_bias[ids]+d.qfrc_constraint[ids]-wrong
                previous=d.qfrc_actuator[ids].copy()
                correct,_=bam.compute_frictions(previous,ext,v);incorrect,_=bam.compute_frictions(previous,bad_ext,v)
                tensor=lambda x:torch.tensor(x[None],dtype=torch.float32,device=env.device)
                tv=tensor(v);stribeck=torch.exp(-torch.pow(tv.abs()/tm.dtheta_stribeck.value,tm.alpha.value))
                torch_budget=actuator._compute_friction_budget(tensor(previous).expand(4,-1),tensor(ext).expand(4,-1),stribeck.expand(4,-1))[0].cpu().numpy()
                old_diffs.append(float(np.abs(incorrect-torch_budget).max()));new_diffs.append(float(np.abs(correct-torch_budget).max()))
                # CPU first control dt is 0; subsequent updates use physics dt.
                control=tm.actuator.compute_control(tensor(pose),tensor(q),tv,0. if step==0 else dt)
                torque=tm.actuator.compute_torque(control,True,tensor(q),tv)[0].cpu().numpy()
                controller.update();torque_diffs.append(float(np.abs(torque-d.ctrl[controller.act_indexes]).max()))
                samples.append({'sit':sit,'step':step,'frictionForceMaxNm':float(np.abs(f).max()),'oldBudgetMaxDifferenceNm':old_diffs[-1]})
                mujoco.mj_step(cm,d)
        report={'hardwareAccess':False,'formalAccepted':False,'scope':'480 frozen-load friction/torque samples from CPU integration plus named assembled model fields; not complete CPU/Warp rollout equivalence.',
            'namedModelFieldChecks':checks,'namedModelMismatches':mismatches,
            'maxOldFrictionBudgetDifferenceNm':max(old_diffs),'maxCorrectedFrictionBudgetDifferenceNm':max(new_diffs),
            'maxNumpyTorchMotorTorqueDifferenceNm':max(torque_diffs),'samples':samples,
            'cpuControllerRevision':cpu._controller_module.FRICTION_CONSTRAINT_REVISION,
            'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'scripts/bam_cpu_controller.py',a.onnx)},
            'gpuPlayActuator':{'delayMin':actuator.cfg.delay_min_lag,'delayMax':actuator.cfg.delay_max_lag,'physicsDt':actuator._dt}}
        a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
