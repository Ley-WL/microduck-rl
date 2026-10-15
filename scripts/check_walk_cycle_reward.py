"""Recompute both tracking-bias scores from recorded physical velocities."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
data = json.loads((a.evidence / 'reward-audit.json').read_text())
windows = [(1., 3.), (5., 11.), (11.3, 16.), (16., 18.), (20., 26.), (26.3, 30.)]
reports = []
for row in data['results']:
    with np.load(a.evidence / row['motionTrace']) as archive:
        values = archive['values'].astype(np.float64)
        dt = float(archive['dt'])
    assert values.shape[0] == 1500 and abs(dt - .02) < 1e-12
    keep = np.asarray(row['survival'], dtype=bool)
    assert keep.any()
    assert 'biasInputTrace' in row, 'Reward-time input capture required; post-forward motion is a different sensor view'
    with np.load(a.evidence / row['biasInputTrace']) as archive:
        inputs = archive['values'].astype(np.float64)
    assert inputs.shape == (1500, values.shape[1], 6)
    actual = inputs[:, keep, :3]
    # mjlab computes reward before CommandManager.compute. The trace records
    # post-step commands; reset explicitly sets the first command to zero.
    command = inputs[:, keep, 3:6]
    np.testing.assert_array_equal(command, np.concatenate((np.zeros_like(values[:1, keep, 3:6]), values[:-1, keep, 3:6])))
    old_error = np.zeros_like(actual[0]); new_error = old_error.copy()
    old_rates = []; new_rates = []
    for velocity, target in zip(actual, command):
        speed = np.linalg.norm(target[:, :2], axis=-1)
        tau = np.where(speed > .001, np.clip(.1 / np.maximum(speed, .025), .7, 4.), .7)
        old_error += (velocity - target - old_error) * dt / (.7 + dt)
        new_error += (velocity - target - new_error) * dt / (tau[:, None] + dt)
        old_rates.append(-4 * np.minimum(((old_error / [.1, .1, .4]) ** 2).sum(-1), 25.))
        new_rates.append(-4 * np.minimum(((new_error / [.1, .1, .4]) ** 2).sum(-1), 25.))
    audit = row['rewardAudit']; terms = audit['terms']; index = terms.index('tracking_bias')
    t = (np.arange(1500) + 1) * dt
    errors = []
    for rates, key in [(old_rates, 'meanLegacyWeightedRate'), (new_rates, 'meanWeightedRate')]:
        rates = np.asarray(rates)
        computed = np.stack([rates[(t >= start) & (t < end)].mean(0) for start, end in windows], axis=1)
        stored = np.asarray(audit[key])[keep, :, index]
        np.testing.assert_allclose(computed, stored, atol=2e-5, rtol=1e-5)
        errors.append(float(abs(computed - stored).max()))
    unchanged = [i for i, name in enumerate(terms) if name not in
                 ('tracking_bias', 'track_linear_velocity', 'track_angular_velocity')]
    assert np.array_equal(np.asarray(audit['meanWeightedRate'])[:, :, unchanged],
                          np.asarray(audit['meanLegacyWeightedRate'])[:, :, unchanged])
    reports.append(dict(policy=row['policy'], seed=row['seed'], checkedWorlds=int(keep.sum()),
                        excludedFailedWorlds=int((~keep).sum()), maxLegacyAndNewError=errors,
                        rewardVsPostForwardMeanAbsoluteDifferenceVxVyYaw=abs(actual-values[:,keep,:3]).mean((0,1)).tolist()))
a.output.write_text(json.dumps(dict(passed=True, reports=reports, optimizationSteps=0,
                                    gaitAccepted=False), indent=2), encoding='utf-8')
print(json.dumps(reports, indent=2))
