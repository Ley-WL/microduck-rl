"""Direction-resolved paired screen; fixed grid, never rewards post-fall resets."""
import argparse, hashlib, json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab_microduck.tasks.mdp import quat_apply
from mjlab_microduck.tasks.walk_tracking_transfer import make_cfg, rl_cfg

p = argparse.ArgumentParser()
p.add_argument('--seeds', type=int, nargs='+', default=[191051, 191057])
p.add_argument('--output', default='tracking-screen.json')
a = p.parse_args()
ROOT = Path(__file__).resolve().parents[2]
cases = [(v, yaw) for v in (-.2, -.1, 0., .1, .2) for yaw in (-.6, 0., .6)]
rows = []
for seed in a.seeds:
    starts = None
    for label, checkpoint in [('baseline', 'inputs/seed.pt'), ('candidate', 'evidence/main/final.pt')]:
        torch.manual_seed(seed)
        cfg = make_cfg(); cfg.scene.num_envs = 60; cfg.episode_length_s = 26.
        cfg.observations['actor'].enable_corruption = False
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        try:
            agent = asdict(rl_cfg()); agent['upload_model'] = False
            runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), agent, log_dir=None, device=env.device)
            runner.load(str(ROOT / checkpoint), load_cfg={'actor': True}, map_location=env.device)
            actor = runner.alg.actor; actor.eval(); obs, _ = env.reset(seed=seed)
            robot = env.scene['robot']; feet, _ = robot.find_sites(['left_foot', 'right_foot'])
            sensor = env.scene['feet_ground_contact']
            initial = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                robot.data.joint_pos, robot.data.joint_vel], dim=-1).cpu()
            if starts is None: starts = initial.clone()
            else: assert torch.equal(starts, initial), 'Unpaired initial states'
            term = env.command_manager.get_term('twist')
            grid = torch.tensor(cases, device=env.device).repeat_interleave(4, dim=0)
            term.stop_speed[:] = grid[:, 0]; term.stop_yaw[:] = grid[:, 1]
            failed = torch.zeros(60, device=env.device, dtype=torch.bool)
            peak = torch.zeros(60, device=env.device)
            sums = torch.zeros((60, 2, 8), device=env.device)
            counts = torch.zeros((60, 2), device=env.device)
            stop = torch.zeros((60, 2), device=env.device); stopcount = torch.zeros(60, device=env.device)
            lift_xy = torch.zeros((60, 2, 2), device=env.device)
            lift_dir = torch.zeros_like(lift_xy); valid = torch.zeros((60, 2), device=env.device, dtype=torch.bool)
            steps = torch.zeros((60, 2), device=env.device); distance = torch.zeros_like(steps)
            airsum = torch.zeros_like(steps)
            command_hash = hashlib.sha256()
            for tick in range(1200):
                with torch.inference_mode(): action = actor(obs)
                obs, _, done, timeout, _ = env.step(action)
                tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))
                peak = torch.maximum(peak, torch.where(~failed, tilt, 0.))
                failed |= done | timeout | (tilt >= 45.)
                live = ~failed
                command = env.command_manager.get_command('twist')
                # Failed worlds are excluded permanently; compare only original commands.
                command_hash.update(command.cpu().numpy().tobytes())
                phase = ((tick + 1) * .02) % 12.
                actual = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), -1)
                if 2. <= phase < 4. or 8. <= phase < 10.:
                    direction = 0 if phase < 7. else 1
                    values = torch.cat((command[:, 0:1], actual[:, 0:1], command[:, 2:3], actual[:, 2:3],
                        (actual[:, 0:1] - command[:, 0:1]).abs(),
                        (actual[:, 2:3] - command[:, 2:3]).abs(), actual[:, 1:2].abs(), tilt[:, None]), -1)
                    sums[:, direction] += torch.where(live[:, None], values, 0.)
                    counts[:, direction] += live
                if 4.3 <= phase < 7. or 10.3 <= phase < 12.:
                    stop += torch.where(live[:, None], torch.stack((actual[:, :2].norm(dim=-1), actual[:, 2].abs()), -1), 0.)
                    stopcount += live
                xy = robot.data.site_pos_w[:, feet, :2]
                lift = sensor.compute_first_air(dt=env.step_dt) & live[:, None]
                velocity = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), -1)
                world = quat_apply(robot.data.root_link_quat_w, velocity)[:, :2]
                heading = world / world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                lift_xy = torch.where(lift[:, :, None], xy, lift_xy)
                lift_dir = torch.where(lift[:, :, None], heading[:, None], lift_dir)
                valid |= lift
                landing = sensor.compute_first_contact(dt=env.step_dt) & valid & live[:, None]
                valid &= ~landing
                landing &= (sensor.data.last_air_time >= .04) & (command[:, 0].abs() > .03)[:, None]
                progress = ((xy - lift_xy) * lift_dir).sum(-1)
                # Retain signed travel: dragging the foot backwards is not a long step.
                steps += landing
                distance += torch.where(landing, progress, 0.)
                airsum += torch.where(landing, sensor.data.last_air_time, 0.)
            metrics = (sums / counts.clamp_min(1)[:, :, None]).cpu().tolist()
            row = dict(policy=label, seed=seed, survival=live.cpu().tolist(), maxTiltDeg=peak.cpu().tolist(),
                metrics=metrics, metricColumns=['commandVx', 'actualVx', 'commandYaw', 'actualYaw', 'vxMAE', 'yawMAE', 'absVy', 'tilt'],
                stop=(stop / stopcount.clamp_min(1)[:, None]).cpu().tolist(),
                stepCount=steps.cpu().tolist(), stepDistance=(distance / steps.clamp_min(1)).cpu().tolist(),
                stepAirTime=(airsum / steps.clamp_min(1)).cpu().tolist(),
                initialStateSha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                commandTraceSha256=command_hash.hexdigest())
            rows.append(row)
            print(label, seed, 'survived', int(live.sum()), flush=True)
            (ROOT / 'evidence' / a.output).write_text(json.dumps(dict(cases=cases, results=rows, accepted=False), indent=2))
        finally:
            env.close()
comparisons = []
for seed in a.seeds:
    b, c = [r for r in rows if r['seed'] == seed]
    keep = np.array(b['survival']) & np.array(c['survival'])
    lost = [i for i in range(60) if b['survival'][i] and not c['survival'][i]]
    per_case = []
    for index, case in enumerate(cases):
        ids = np.arange(index * 4, index * 4 + 4); ids = ids[keep[ids]]
        if not len(ids):
            per_case.append(dict(command=case, paired=0)); continue
        values = {}
        for label, row in [('baseline', b), ('candidate', c)]:
            m = np.array(row['metrics'])[ids].mean(0)
            values[label] = dict(phases=m.tolist(), vxMAE=float(m[:, 4].mean()), yawMAE=float(m[:, 5].mean()),
                stop=np.array(row['stop'])[ids].mean(0).tolist(),
                stepDistance=float(np.array(row['stepDistance'])[ids].mean()),
                stepCount=float(np.array(row['stepCount'])[ids].sum(1).mean()))
        per_case.append(dict(command=case, paired=len(ids), **values))
    # Per-direction regressions cannot be hidden by the overall average.
    directional = all(r['paired'] and r['candidate']['vxMAE'] <= r['baseline']['vxMAE'] * 1.05 + .005
        and r['candidate']['yawMAE'] <= r['baseline']['yawMAE'] * 1.05 + .02
        and r['candidate']['stop'][0] <= r['baseline']['stop'][0] * 1.1 + .002 for r in per_case)
    aggregate = {}
    for label, row in [('baseline', b), ('candidate', c)]:
        m = np.array(row['metrics'])[keep]
        walking = keep & (np.repeat(np.array(cases)[:, 0], 4) != 0)
        aggregate[label] = dict(vxMAE=float(m[:, :, 4].mean()) if keep.any() else 1e9,
            yawMAE=float(m[:, :, 5].mean()) if keep.any() else 1e9,
            stepDistance=float(np.array(row['stepDistance'])[walking].mean()) if walking.any() else 0.)
    bb, cc = aggregate['baseline'], aggregate['candidate']
    passed = not lost and keep.sum() >= 56 and directional and cc['vxMAE'] <= .9 * bb['vxMAE'] and cc['yawMAE'] <= .9 * bb['yawMAE'] and cc['stepDistance'] >= 1.1 * bb['stepDistance']
    comparisons.append(dict(seed=seed, lost=lost, paired=int(keep.sum()), perCase=per_case,
        aggregate=aggregate, directionRegressionPassed=bool(directional), passed=bool(passed)))
result = dict(cases=cases, results=rows, comparisons=comparisons, accepted=all(r['passed'] for r in comparisons),
    deploymentAccepted=False, hardwareValidated=False, simulationOnly=True,
    boundary='same initial physical states; post-reset failed worlds excluded; physics randomization not asserted bitwise paired')
(ROOT / 'evidence' / a.output).write_text(json.dumps(result, indent=2))
print('SCREEN', result['accepted'], [(r['seed'], r['aggregate']) for r in comparisons], flush=True)
