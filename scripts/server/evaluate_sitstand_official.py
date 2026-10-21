"""Paired nominal screening: both repeated cycles and both holds, no hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.sitstand_observer_official import make_cfg
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    policy = ort.InferenceSession(args.policy, sess_options=options, providers=['CPUExecutionProvider'])
    reports = {}
    for scenario in ('stand_cycle', 'sit_cycle', 'stand_hold', 'sit_hold'):
        seed = 211127
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = make_cfg(play=True)
        cfg.seed = seed
        cfg.scene.num_envs = args.envs
        cfg.episode_length_s = 26
        cfg.curriculum.clear()
        for key in ('push_robot', 'randomize_com', 'randomize_head_com', 'randomize_mass_inertia', 'randomize_armature', 'randomize_joint_friction', 'encoder_bias', 'imu_orientation'):
            cfg.events.pop(key, None)
        sitting = scenario.startswith('sit_')
        cfg.events['set_ground_state'].params.update(sitting_prob=float(sitting), standing_prob=float(not sitting))
        cfg.commands['head_pose'].ranges = tuple((0., 0.) for _ in cfg.commands['head_pose'].ranges)
        cfg.observations['actor'].enable_corruption = False
        cfg.observations['actor'].terms['joint_pos'].params['biased'] = False
        for actuator in cfg.scene.entities['robot'].articulation.actuators:
            actuator.vin_range = (7.4, 7.4)
            actuator.vin_drop_gain_range = (0., 0.)
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        try:
            obs, _ = env.reset(seed=seed)
            initial = hashlib.sha256(env.sim.data.qpos.cpu().numpy().tobytes() + env.sim.data.qvel.cpu().numpy().tobytes()).hexdigest()
            robot = env.scene['robot']
            term = env.command_manager.get_term('twist')
            bad = torch.zeros(args.envs, dtype=torch.bool, device=env.device)
            good, traces, qposes = [], [], []
            home = robot.data.default_joint_pos.clone()
            sit = home.clone()
            for index, value in SITTING_TARGET_OVERRIDES.items():
                sit[:, index] = value
            legs = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
            for tick in range(1200):
                segment = tick // 300
                flag = float(sitting if scenario.endswith('hold') else (sitting if segment % 2 == 0 else not sitting))
                term.vel_command_b.zero_()
                term.vel_command_b[:, 0] = flag
                term.time_left.fill_(100.)
                obs['actor'][:, 48:51] = term.vel_command_b
                inputs = obs['actor'].cpu().numpy()
                action = np.concatenate([policy.run(None, {policy.get_inputs()[0].name: row[None]})[0] for row in inputs])
                if not np.isfinite(action).all():
                    raise RuntimeError('Non-finite policy action')
                with torch.no_grad():
                    obs, _, done, timeout, _ = env.step(torch.as_tensor(action, device=env.device))
                tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
                height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                bad |= done | timeout | (tilt > 1.0472)
                target = sit if flag else home
                error = (robot.data.joint_pos[:, legs] - target[:, legs]).abs().mean(-1)
                stable = (tilt < .3491) & ((height - (.06 if flag else .115)).abs() < .012) & (error < .25) & ~bad
                good.append(stable.cpu().numpy())
                traces.append(torch.stack((height, tilt, error), -1).cpu().numpy())
                if tick % 2 == 0:
                    qposes.append(env.sim.data.qpos[0].cpu().numpy().copy())
            frames = np.asarray(good)
            fractions = np.asarray([frames[end-100:end].mean(0) for end in (300, 600, 900, 1200)])
            success = (fractions >= .95).all(0)
            reports[scenario] = dict(success=int(success.sum()), survived=int((~bad).sum()), worlds=args.envs,
                endpointFractions=fractions.tolist(), initialQposQvelSha256=initial,
                policySha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(), screeningOnly=True)
            (out/'results.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
            np.savez_compressed(out/(scenario+'.npz'), stable=frames, trajectory=np.asarray(traces), qpos=np.asarray(qposes))
            print(scenario, reports[scenario], flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--envs', type=int, default=16)
    run(parser.parse_args())
