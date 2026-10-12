"""Check GPU BAM standing/seated equilibria before blaming policy learning."""
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_hd1910_skills_env_cfg import (
    SIT_Z, STAND_Z, SITTING_TARGET_OVERRIDES,
)


def run(profile, output, task='Mjlab-SitStand-Flat-MicroDuck-HD1910'):
    cfg = load_env_cfg(task, play=profile.startswith('nominal'))
    sit_overrides = cfg.events['set_ground_state'].params['sit_overrides']
    sit_z,stand_z = cfg.commands['twist'].sit_z,cfg.commands['twist'].stand_z
    cfg.scene.num_envs = 64
    cfg.curriculum.clear()
    cfg.episode_length_s = 10000.
    cfg.terminations.pop('time_out', None)
    cfg.commands['twist'].resampling_time_range = (10000., 10000.)
    if profile.endswith('clean'):
        cfg.events['set_ground_state'].params.update(joint_noise=0., tilt_noise_deg=0.)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        env.reset(seed=5109)
        robot = env.scene['robot']
        home = robot.data.default_joint_pos.clone()
        knee = robot.joint_names.index('left_knee')
        is_sit = robot.data.joint_pos[:, knee] > .7
        target = home.clone()
        for index, value in sit_overrides.items():
            target[is_sit, index] = value
        action = target - home
        action_term = env.action_manager.get_term('joint_pos')
        initial = robot.data.joint_pos.clone()
        if profile.endswith(('aligned', 'clean')):
            action_term._filtered_actions.copy_(action)
        twist = env.command_manager.get_term('twist')
        twist.vel_command_b[:] = 0
        twist.vel_command_b[:, 0] = is_sit.float()
        twist._alpha.copy_(is_sit.float())
        samples = []
        rewards = []
        first_command = None
        for tick in range(round(5. / env.step_dt)):
            _, reward, terminated, truncated, _ = env.step(action)
            if first_command is None:
                first_command = action_term._processed_actions.cpu().numpy().tolist()
            if terminated.any() or truncated.any():
                raise RuntimeError('Pose hold terminated/reset')
            if tick >= round(4. / env.step_dt):
                tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.)))
                height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                error = (robot.data.joint_pos - target).abs().mean(-1)
                samples.append(torch.stack((tilt, height, error), dim=-1).cpu().numpy())
                rewards.append(reward.cpu().numpy() / env.step_dt)
        array = np.array(samples)
        groups = {}
        for label, mask, height in [('stand', ~is_sit, stand_z), ('sit', is_sit, sit_z)]:
            mask = mask.cpu().numpy()
            values = array[:, mask]
            valid = (values[:, :, 0] < 20.) & (abs(values[:, :, 1]-height) < .012)
            groups[label] = {'worlds': int(mask.sum()),
                             'uprightHeightStableWorlds': int((valid.mean(0) >= .95).sum()),
                             'tiltP95Deg': float(np.percentile(values[:, :, 0], 95)),
                             'heightMeanM': float(values[:, :, 1].mean()),
                             'jointErrorMeanRad': float(values[:, :, 2].mean()),
                             'rewardRateMean': float(np.array(rewards)[:, mask].mean())}
        mujoco.mj_saveModel(env.sim.mj_model, str(output.parent / (profile + '-compiled.mjb')), None)
        return {'profile': profile, 'groups': groups, 'acceptanceEvidence': False,
                'jointNames': robot.joint_names, 'initialQ': initial.cpu().numpy().tolist(),
                'bodyGroundContactMasks': [
                    {'name': env.sim.mj_model.geom(i).name,
                     'contype': int(env.sim.mj_model.geom_contype[i]),
                     'conaffinity': int(env.sim.mj_model.geom_conaffinity[i])}
                    for i in range(env.sim.mj_model.ngeom)
                    if env.sim.mj_model.geom(i).name.startswith('robot/body_ground_')],
                'expectedTargets': target.cpu().numpy().tolist(), 'firstCommand': first_command,
                'finalQ': robot.data.joint_pos.cpu().numpy().tolist()}
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='Mjlab-SitStand-Flat-MicroDuck-HD1910')
    p.add_argument('--profiles', nargs='+', default=['nominal-aligned', 'nominal-clean', 'training-aligned'],
                   choices=['nominal', 'training', 'nominal-aligned', 'nominal-clean', 'training-aligned'])
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = [run(profile, args.output, args.task) for profile in args.profiles]
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps([{k: row[k] for k in ('profile', 'groups')} for row in result]))


if __name__ == '__main__':
    main()
