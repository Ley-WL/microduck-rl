"""Offline low-speed relative-error scoring on frozen reward-time trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = a.evidence / 'reward-audit.json'
data = json.loads(source.read_text())
windows = [(1., 3.), (5., 11.), (11.3, 16.), (16., 18.), (20., 26.), (26.3, 30.)]
for row in data['results']:
    assert all(row['survival']), 'This replay requires complete trajectories; never include reset rewards'
    with np.load(a.evidence / row['biasInputTrace']) as archive:
        inputs = archive['values'].astype(np.float64)
        dt = float(archive['dt'])
    error = np.zeros_like(inputs[0, :, :3]); rates = []; preserved = []
    for frame in inputs:
        actual, command = frame[:, :3], frame[:, 3:]
        speed = np.linalg.norm(command[:, :2], axis=-1)
        tau = np.where(speed > .001, np.clip(.1 / np.maximum(speed, .025), .7, 4.), .7)
        error += (actual - command - error) * dt / (tau[:, None] + dt)
        vx = abs(command[:, 0])
        scale = np.tile([.1, .1, .4], (len(frame), 1))
        scale[:, 0] = np.where(vx > .001, np.clip(vx, .025, .1), .1)
        rates.append(-4 * np.minimum(((error / scale) ** 2).sum(-1), 25.))
        preserved.append(-4 * np.minimum(((error / [.1, .1, .4]) ** 2).sum(-1), 25.))
    rates = np.asarray(rates); preserved = np.asarray(preserved)
    vx = abs(inputs[:, :, 3]); unchanged = (vx <= .001) | (vx >= .1)
    np.testing.assert_array_equal(rates[unchanged], preserved[unchanged])
    audit = row['rewardAudit']; index = audit['terms'].index('tracking_bias')
    t = (np.arange(len(inputs)) + 1) * dt
    computed = np.stack([rates[(t >= start) & (t < end)].mean(0) for start, end in windows], axis=1)
    original = np.stack([preserved[(t >= start) & (t < end)].mean(0) for start, end in windows], axis=1)
    weighted = np.asarray(audit['meanWeightedRate'])
    np.testing.assert_allclose(original, weighted[:, :, index], atol=2e-5, rtol=1e-5)
    audit['meanCycleWeightedRate'] = weighted.tolist()
    weighted[:, :, index] = computed
    audit['meanWeightedRate'] = weighted.tolist()
    audit['task'] = 'offline_relative_longitudinal_error'
data['offlineRewardReplay'] = dict(sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    training=False, parametersUpdated=False, rule='Cycle averaging plus vx error scale clamp(abs(commandVx),.025,.1); zero/pure-yaw scale .1',
    limitation='In-sample reward hypothesis, not a trained actor or independent gait validation')
a.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
print(a.output)
