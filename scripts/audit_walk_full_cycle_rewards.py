"""Validate full reward traces and decompose all start/stop frames; no optimization."""
import argparse
import json
from pathlib import Path

import numpy as np


def relative_bias(inputs, dt):
    error = np.zeros_like(inputs[0, :, :3], dtype=np.float64)
    rates = []
    for frame in inputs:
        target = frame[:, 3:]
        speed = np.linalg.norm(target[:, :2], axis=-1)
        tau = np.where(speed > .001, np.clip(.1 / np.maximum(speed, .025), .7, 4.), .7)
        error += (frame[:, :3] - target - error) * dt / (tau[:, None] + dt)
        scale = np.tile([.1, .1, .4], (len(target), 1))
        scale[:, 0] = np.where(abs(target[:, 0]) > .001, np.clip(abs(target[:, 0]), .025, .1), .1)
        rates.append(-4 * np.minimum(((error / scale) ** 2).sum(-1), 25.))
    return np.asarray(rates)


def audit(root):
    data = json.loads((root / 'reward-audit.json').read_text())
    loaded = {}
    checks = []
    for row in data['results']:
        assert row['rewardAudit']['task'] == 'walk_relative_cycle_context'
        with np.load(root / row['rewardTrace']) as z:
            rates = z['weightedRates'].astype(np.float64)
            rewards = z['scalarRewards'].astype(np.float64)
            terms = z['terms'].tolist()
            scale = float(z['scale']); dt = float(z['dt'])
        with np.load(root / row['biasInputTrace']) as z:
            inputs = z['values'].astype(np.float64)
        with np.load(root / row['motionTrace']) as z:
            motion = z['values'].astype(np.float64)
        assert rates.shape == (1500, len(row['survival']), len(terms))
        assert inputs.shape == (1500, rates.shape[1], 6)
        assert rewards.shape == rates.shape[:2] and abs(dt - .02) < 1e-12
        assert terms == row['rewardAudit']['terms'] and np.isfinite(rates).all()
        live = motion[:, :, -1].astype(bool)
        assert not np.any(live[1:] & ~live[:-1]), 'Failed worlds cannot rejoin'
        keep = np.asarray(row['survival'], dtype=bool)
        assert np.array_equal(live[-1], keep)
        np.testing.assert_allclose(rates.sum(-1)[live] * scale, rewards[live], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(inputs[:, keep, 3:],
            np.concatenate((np.zeros_like(motion[:1, keep, 3:6]), motion[:-1, keep, 3:6])))
        replay = relative_bias(inputs[:, keep], dt)
        stored = rates[:, keep, terms.index('tracking_bias')]
        np.testing.assert_allclose(replay, stored, atol=2e-5, rtol=1e-5)
        checks.append(dict(seed=row['seed'], policy=row['policy'], survived=int(keep.sum()),
            worlds=len(keep), maxBiasReplayError=float(abs(replay-stored).max()),
            maxRewardSumError=float(abs(rates.sum(-1)[live]*scale-rewards[live]).max())))
        loaded[row['seed'], row['policy']] = (row, rates, inputs, terms, dt, scale)
    # Exhaustive frame partitions, unlike the older six disjoint windows.
    bounds = [0, 50, 150, 550, 565, 800, 900, 1300, 1315, 1500]
    names = ['initialRest','startForward','walkForward','brakeFirst','holdFirst',
             'startReverse','walkReverse','brakeSecond','holdSecond']
    pairs = []
    failed = []
    for seed in sorted({key[0] for key in loaded}):
        b, c = loaded[seed, 'baseline'], loaded[seed, 'candidate']
        assert b[0]['initialStateSha256'] == c[0]['initialStateSha256']
        assert b[3:] == c[3:]
        for world, (blive, clive) in enumerate(zip(b[0]['survival'], c[0]['survival'])):
            entry = dict(seed=seed, world=world, command=data['cases'][world//4],
                baselineSurvived=blive, candidateSurvived=clive)
            if not (blive and clive):
                failed.append(entry)
                continue
            np.testing.assert_array_equal(b[2][:, world, 3:], c[2][:, world, 3:])
            for label, source in [('baseline', b), ('candidate', c)]:
                rates, inputs, terms, dt, scale = source[1:]
                r = rates[:, world]
                phase = []
                for name, lo, hi in zip(names, bounds[:-1], bounds[1:]):
                    phase.append(dict(name=name, frames=hi-lo,
                        meanRate=r[lo:hi].mean(0).tolist(),
                        integrated=(r[lo:hi].sum(0)*dt).tolist(),
                        meanActual=inputs[lo:hi, world, :3].mean(0).tolist(),
                        meanCommand=inputs[lo:hi, world, 3:].mean(0).tolist()))
                total = r.sum(0)*dt
                np.testing.assert_allclose(np.sum([p['integrated'] for p in phase], axis=0), total, atol=1e-10)
                # Gamma .99 is the training setting. .997 only diagnoses sensitivity;
                # both are truncated recorded returns, not critic readiness targets.
                discounted = {str(gamma): {str(start):
                    ((r[start:]*scale)*gamma**np.arange(len(r)-start)[:, None]).sum(0).tolist()
                    for start in [0, 50, 800]} for gamma in [.99, .997]}
                entry[label] = dict(phases=phase, integrated=total.tolist(), discounted=discounted)
            pairs.append(entry)
    return dict(passed=True, accepted=False, simulationOnly=True, optimizationSteps=0,
        terms=terms, checks=checks, failed=failed, pairs=pairs,
        limitation='Frozen deterministic rollouts; reward preference is diagnostic, not proof of PPO causality. Failed worlds retained separately; finite 30s returns have no bootstrap.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    report = audit(a.evidence)
    a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('passed','checks','failed')}))
