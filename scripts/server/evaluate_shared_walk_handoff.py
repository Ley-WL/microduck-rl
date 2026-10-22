"""Roll 1.6s -> early tuck target -> IMU-gated getup -> walk, nominal screening.

Uses HD1910/Observer roll physics at 7.4V; no hardware I/O or training.
Getup raw-history reset and EMA/filter contract match native 50Hz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.pick_roll_observer_official import make_cfg

DANCE_KEYS=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.04, 0.25, -0.06, 0.0, 0.006, -0.002, 0.055, 0.0, 0.0], [0.0, 0.04, -0.25, 0.06, 0.0, -0.006, -0.002, -0.055, 0.0, 0.0], [0.0, 0.04, 0.25, -0.06, 0.0, 0.006, -0.002, 0.055, 0.0, 0.0], [0.0, 0.04, -0.25, 0.06, 0.0, -0.006, -0.002, -0.055, 0.0, 0.0], [0.1, -0.16, 0.22, -0.05, 0.0, 0.006, -0.006, 0.05, 0.0, 0.0], [-0.04, 0.08, -0.22, 0.05, 0.0, -0.006, 0.0, -0.05, 0.0, 0.0], [0.1, -0.16, 0.22, -0.05, 0.0, 0.006, -0.006, 0.05, 0.0, 0.0], [-0.04, 0.08, -0.22, 0.05, 0.0, -0.006, 0.0, -0.05, 0.0, 0.0], [0.1, -0.16, 0.0, 0.0, 0.0, 0.0, -0.006, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=np.float32)
DANCE_GAINS=np.array([2, 2, 2, 2.5, 1, 2, 2.5, 2, 1, 1],dtype=np.float32)

def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    policy = ort.InferenceSession(args.policy, sess_options=options, providers=['CPUExecutionProvider'])
    walk = ort.InferenceSession(args.walk, sess_options=options, providers=['CPUExecutionProvider'])
    getup = ort.InferenceSession(args.getup, sess_options=options, providers=["CPUExecutionProvider"])
    skill=ort.InferenceSession(args.skill_policy,sess_options=options,providers=["CPUExecutionProvider"])
    reports = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = make_cfg(args.task, play=True)
        cfg.seed = seed
        cfg.scene.num_envs = args.envs
        cfg.episode_length_s = 19
        cfg.curriculum.clear()
        for key in ('push_robot', 'randomize_com', 'randomize_head_com', 'randomize_mass_inertia',
                    'randomize_armature', 'randomize_joint_friction', 'encoder_bias', 'imu_orientation'):
            cfg.events.pop(key, None)
        cfg.observations['actor'].enable_corruption = False
        cfg.observations['actor'].terms['joint_pos'].params['biased'] = False
        for actuator in cfg.scene.entities['robot'].articulation.actuators:
            actuator.vin_range = (7.4, 7.4)
            actuator.vin_drop_gain_range = (0., 0.)
        if args.task == 'roll':
            cfg.events['set_roulade_state'].params.update(standing_prob=1., midroll_prob=0.)
        else:
            cfg.commands['twist'].randomize_phase = False
            # Nominal screening has no payload; training keeps the official 10-40g.
            cfg.events['sample_mouth_payload'].params.update(min_kg=0., max_kg=0.)
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        try:
            obs, _ = env.reset(seed=seed)
            initial = hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes() + env.sim.data.qvel.cpu().numpy().tobytes()).hexdigest()
            robot = env.scene['robot']
            term = env.command_manager.get_term('twist')
            bad = torch.zeros(args.envs, dtype=torch.bool, device=env.device)
            stable_frames, task_frames, qposes = [], [], []
            takeover_states=[None for _ in range(args.envs)]
            quiet_frames=[]; leg_indices=np.array([0,1,2,3,4,9,10,11,12,13])
            previous_phase=None; rotation=np.zeros(args.envs); inverted=np.zeros(args.envs,dtype=bool); run=np.zeros(args.envs,dtype=int); switched=np.zeros(args.envs,dtype=bool); switched_at=np.full(args.envs,-1.); previous_action=np.zeros((args.envs,14),dtype=np.float32); anchor=previous_action.copy()
            ticks = 900
            if args.task == 'pick':
                site_id = robot.find_sites('mouth_tip')[0][0]
            phase_state=np.zeros(args.envs,dtype=int); dwell=np.zeros(args.envs,dtype=int); frozen=np.zeros((args.envs,14),dtype=np.float32); phase_log=[]; getup_at=np.full(args.envs,-1); raw_history=np.zeros((args.envs,14),dtype=np.float32); filtered=raw_history.copy(); tuck=raw_history.copy(); sensor_velocity=raw_history.copy(); sensor_gyro=np.zeros((args.envs,3),dtype=np.float32)
            for tick in range(ticks):
                if args.task == 'pick':
                    phase = (tick % 200) / 200
                    term._gp_phase.fill_(phase)
                    term.vel_command_b[:, 0] = np.cos(2 * np.pi * phase)
                    term.vel_command_b[:, 1] = np.sin(2 * np.pi * phase)
                    term.vel_command_b[:, 2] = 0
                else:
                    term.vel_command_b.zero_()
                    term.time_left.fill_(100.)
                obs['actor'][:, 48:51] = term.vel_command_b
                rows=obs['actor'].cpu().numpy()
                rows[:,48:61]=0
                if args.skill=='pick': rows[phase_state==0,48]=np.cos(2*np.pi*tick*.02/4);rows[phase_state==0,49]=np.sin(2*np.pi*tick*.02/4)
                if args.skill=='sitstand':rows[phase_state==0,48]=float(tick<200)
                if args.skill=='flamingo':rows[phase_state==0,48]=float(tick<150);rows[phase_state==0,49]=-1.
                if args.skill=='dance':
                    beat=np.clip(((tick*.02)%8-1)/.6,0,10);i=min(int(beat),9);u=beat-i;blend=u*u*u*(10-15*u+6*u*u)
                    rows[phase_state==0,51:61]=(DANCE_KEYS[i]+blend*(DANCE_KEYS[i+1]-DANCE_KEYS[i]))*DANCE_GAINS
                # Reproduce the native 50Hz getup filters. Donor skills keep
                # their configured one-frame joint velocity delay.
                vterm=cfg.observations['actor'].terms['joint_vel']
                gterm=cfg.observations['actor'].terms['base_ang_vel']
                vnow=vterm.func(env,**vterm.params).cpu().numpy()
                gnow=gterm.func(env,**gterm.params).cpu().numpy()
                recovering=(phase_state==2)|((phase_state==0)&(args.skill=="dance"))
                sensor_velocity=np.where(recovering[:,None],.16*sensor_velocity+.84*vnow,vnow)
                sensor_gyro=np.where(recovering[:,None],.25*sensor_gyro+.75*gnow,gnow)
                rows[recovering,20:34]=sensor_velocity[recovering]
                rows[recovering,:3]=sensor_gyro[recovering]
                g=rows[:,3:6]; norm=np.linalg.norm(g,axis=1)
                tilt_obs=np.degrees(np.arccos(np.clip(-g[:,2]/norm,-1,1))); gyro=np.linalg.norm(rows[:,:3],axis=1)
                stood=(tilt_obs<10)&(gyro<.5)&(np.max(np.abs(rows[:,6:20][:,leg_indices]),axis=1)<.30)&(np.max(np.abs(rows[:,20:34][:,leg_indices]),axis=1)<1)
                dwell=np.where(stood,dwell+1,0)
                certified=dwell>=26
                duration={'pick':2.8,'kick_left':.5,'kick_right':.5,'dance':8.,'flamingo':4.5,'sitstand':4.}[args.skill]
                phase_done=tick*.02>=duration
                end=(phase_state==0)&phase_done
                recovered=(phase_state==2)&certified
                direct=end&certified
                bridge=end&~certified&(args.skill!='sitstand')
                phase_state[bridge]=2;getup_at[bridge]=tick;raw_history[bridge]=0;filtered[bridge]=0;dwell[bridge]=0
                walked=direct|recovered
                # Donor actors with identical HOME retain raw actions into walk.
                reset=walked&((phase_state==2)|(args.skill=='dance'))
                raw_history[reset]=0;filtered[reset]=0
                phase_state[walked]=3;switched_at[walked]=tick*.02;switched|=walked;dwell[walked]=0
                rows[bridge|walked,48:61]=0
                rows[:,34:48]=raw_history
                action=np.empty((args.envs,14),dtype=np.float32)
                for i,row in enumerate(rows):
                    if phase_state[i]==4: action[i]=frozen[i]
                    elif phase_state[i]==1: action[i]=tuck[i];raw_history[i]=tuck[i]
                    else:
                        net=skill if phase_state[i]==0 else getup if phase_state[i]==2 else walk
                        raw=net.run(None,{net.get_inputs()[0].name:row[None]})[0][0]
                        raw_history[i]=raw
                        filtered[i]=.45*filtered[i]+.55*raw if phase_state[i]==2 or (phase_state[i]==0 and args.skill=="dance") else raw
                        action[i]=filtered[i]
                if tick==args.tuck_tick: tuck=action.copy()
                phase_log.append(phase_state.copy())
                action_step=np.max(np.abs(action[:,leg_indices]-previous_action[:,leg_indices]),axis=1)
                previous_action=action.copy()
                if not np.isfinite(action).all():
                    raise RuntimeError('Non-finite policy action')
                with torch.no_grad():
                    obs, _, done, timeout, _ = env.step(torch.as_tensor(action, device=env.device))
                bad |= done | timeout
                tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
                height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                # Conservative walk fall guard (immediate at 60deg); actual
                # runtime allows 150ms below its 90deg immediate threshold.
                bad |= torch.as_tensor(phase_state==3,device=env.device)&(tilt>=np.deg2rad(60))
                bad |= torch.as_tensor((phase_state==2)&((tick-getup_at)>=500),device=env.device)
                if args.skill=="dance": bad |= torch.as_tensor(phase_state==0,device=env.device)&(tilt>=np.deg2rad(45))
                stable = (tilt < .3491) & ((height - .115).abs() < .02) & ~bad
                if args.task == 'pick':
                    bad |= tilt > 1.0472
                    mouth_z = robot.data.site_pos_w[:, site_id, 2] - env.scene.env_origins[:, 2]
                    q = robot.data.site_quat_w[:, site_id]
                    alignment = -2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
                    task_values = torch.stack((mouth_z, alignment), -1)
                else:
                    task_values = torch.stack((env._roulade_max, env._roulade_head_latch.float()), -1)
                next_rows=obs["actor"].cpu().numpy()
                quiet_frames.append(stable.cpu().numpy() & (np.linalg.norm(next_rows[:,:3],axis=1)<1.5) & (np.max(np.abs(next_rows[:,20:34][:,leg_indices]),axis=1)<4) & (action_step<np.deg2rad(15)))
                stable_frames.append(stable.cpu().numpy())
                task_frames.append(task_values.cpu().numpy())
                qposes.append(env.sim.data.qpos[0].cpu().numpy().copy())
            stable = np.asarray(stable_frames)
            values = np.asarray(task_frames)
            if args.task == 'pick':
                cycles = []
                for start in (0, 200, 400):
                    near = values[start+65:start+90]
                    approach = ((near[:, :, 0] >= -.005) & (near[:, :, 0] < .025)
                                & (near[:, :, 1] > .7)).sum(0) >= 5
                    returned = stable[start+170:start+200].mean(0) >= .9
                    cycles.append(approach & returned)
                success = np.asarray(cycles).all(0) & ~bad.cpu().numpy()
            else:
                success = np.ones(args.envs,dtype=bool)
                success &= (stable[-25:].mean(0) >= .9) & ~bad.cpu().numpy()
            strict=success & (phase_state==3) & (np.asarray(quiet_frames)[-50:].mean(0)>=.9)
            reports[str(seed)] = dict(getupAt=(getup_at*.02).tolist(),finalPhases=phase_state.tolist(),takeoverStates=takeover_states, strictSuccess=int(strict.sum()), switchedCount=int(switched.sum()), settings=vars(args), success=int(success.sum()), successByWorld=success.tolist(),
                finalTiltDegrees=np.degrees(tilt.cpu().numpy()).tolist(), finalHeight=height.cpu().numpy().tolist(), noReset=int((~bad).sum()), switchedAt=switched_at.tolist(), worlds=args.envs, initialQposQvelSha256=initial,
                getupSha256=hashlib.sha256(Path(args.getup).read_bytes()).hexdigest(), walkSha256=hashlib.sha256(Path(args.walk).read_bytes()).hexdigest(), skillSha256=hashlib.sha256(Path(args.skill_policy).read_bytes()).hexdigest(), screeningOnly=True)
            (out/'results.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
            np.savez_compressed(out/f'{seed}.npz', phases=np.asarray(phase_log),stable=stable, task=values, qpos=np.asarray(qposes))
            print(seed, reports[str(seed)], flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('policy','walk','getup','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--envs',type=int,default=16)
    parser.add_argument('--skill',choices=['pick','kick_left','kick_right','dance','flamingo','sitstand'],required=True)
    parser.add_argument('--skill-policy',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[221251,221257])
    parser.set_defaults(task='roll',tuck_tick=5)
    args=parser.parse_args()
    if args.envs<1: parser.error('envs must be positive')
    run(args)
