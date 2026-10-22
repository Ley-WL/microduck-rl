"""Paired nominal skill screening; explicit criteria, not hardware acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.pick_roll_observer_official import make_cfg


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    policy = ort.InferenceSession(args.policy, sess_options=options, providers=['CPUExecutionProvider'])
    reports = {}
    for seed in (211131, 211139):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = make_cfg(args.task, play=True)
        cfg.seed = seed
        cfg.scene.num_envs = args.envs
        cfg.episode_length_s = 14 if args.task == 'pick' else 6
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
            ticks = 600 if args.task == 'pick' else 250
            if args.task == 'pick':
                site_id = robot.find_sites('mouth_tip')[0][0]
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
                action = np.concatenate([policy.run(None, {policy.get_inputs()[0].name: row[None]})[0]
                                         for row in obs['actor'].cpu().numpy()])
                if not np.isfinite(action).all():
                    raise RuntimeError('Non-finite policy action')
                with torch.no_grad():
                    obs, _, done, timeout, _ = env.step(torch.as_tensor(action, device=env.device))
                bad |= done | timeout
                tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
                height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                stable = (tilt < .3491) & ((height - .115).abs() < .02) & ~bad
                if args.task == 'pick':
                    bad |= tilt > 1.0472
                    mouth_z = robot.data.site_pos_w[:, site_id, 2] - env.scene.env_origins[:, 2]
                    q = robot.data.site_quat_w[:, site_id]
                    alignment = -2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
                    task_values = torch.stack((mouth_z, alignment), -1)
                else:
                    task_values = torch.stack((env._roulade_max, env._roulade_head_latch.float()), -1)
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
                success = (values[-1, :, 0] >= np.deg2rad(330)) & (values[-1, :, 1] > .5)
                success &= (stable[-25:].mean(0) >= .9) & ~bad.cpu().numpy()
            reports[str(seed)] = dict(success=int(success.sum()), successByWorld=success.tolist(),
                noReset=int((~bad).sum()), worlds=args.envs, initialQposQvelSha256=initial,
                policySha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(), screeningOnly=True)
            (out/'results.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
            np.savez_compressed(out/f'{seed}.npz', stable=stable, task=values, qpos=np.asarray(qposes))
            print(seed, reports[str(seed)], flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('pick', 'roll'), required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--envs', type=int, default=16)
    run(parser.parse_args())
