"""Measure the trained sitstand policy under the training command sampler.

This diagnostic does not certify success or change a live training process.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import mjlab_microduck.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_env_cfg('Mjlab-SitStand-Flat-MicroDuck-HD1910', play=False)
    cfg.scene.num_envs = 64
    cfg.curriculum.clear()
    cfg.episode_length_s = 10000.
    cfg.terminations.pop('time_out', None)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        obs, _ = env.reset(seed=8191)
        term = env.command_manager.get_term('twist')
        robot = env.scene['robot']
        frames = []
        seen = np.zeros((64, 2), dtype=bool)
        flips = np.zeros(64, dtype=int)
        previous = None
        mismatch = 0.
        for tick in range(round(30. / env.step_dt)):
            flag = term.command[:, 0].cpu().numpy().astype(int)
            seen[np.arange(64), flag] = True
            if previous is not None:
                flips += flag != previous
            previous = flag.copy()
            mismatch = max(mismatch, float((obs['actor'][:, 48:51] - term.command).abs().max()))
            actor = obs['actor'].cpu().numpy()
            action = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0]
                                     for row in actor])
            obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action).to(env.device))
            if terminated.any() or truncated.any():
                raise RuntimeError('Diagnostic encountered reset/termination')
            if tick % 50 == 0:
                height = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu().numpy()
                tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.))).cpu().numpy()
                frames.append({'seconds': tick * env.step_dt,
                               'sitFraction': float(flag.mean()),
                               'blendMean': float(term.alpha.mean()),
                               'heightMean': float(height.mean()), 'tiltMean': float(tilt.mean())})
        result = {'everyWorldSawBothFlags': bool(seen.all()),
                  'flagCoverageByWorld': seen.tolist(), 'flipsByWorld': flips.tolist(),
                  'maxObservationCommandMismatch': mismatch, 'trace': frames,
                  'acceptanceEvidence': False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result[key] for key in
                          ('everyWorldSawBothFlags', 'maxObservationCommandMismatch')}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
