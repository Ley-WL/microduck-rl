"""Reject corrupted totals, shifted commands, and resurrected failed worlds."""
import json
from pathlib import Path
import tempfile

import numpy as np

from audit_walk_full_cycle_rewards import audit


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    terms = ['tracking_bias', 'test_constant']
    inputs = np.zeros((1500, 1, 6))
    motion = np.zeros((1500, 1, 8)); motion[:, :, -1] = 1
    rates = np.zeros((1500, 1, 2)); rates[:, :, 1] = 2
    scalar = np.full((1500, 1), .04)
    def write(r=rates, s=scalar, m=motion, i=inputs):
        np.savez(root/'rewards.npz', weightedRates=r, scalarRewards=s, terms=terms, dt=.02, scale=.02)
        np.savez(root/'motion.npz', values=m)
        np.savez(root/'inputs.npz', values=i)
    row = dict(seed=1, survival=[True], initialStateSha256='identical',
        rewardTrace='rewards.npz', motionTrace='motion.npz', biasInputTrace='inputs.npz',
        rewardAudit=dict(task='walk_relative_cycle_context', terms=terms))
    (root/'reward-audit.json').write_text(json.dumps(dict(cases=[[0,0]], results=[
        dict(row, policy='baseline'), dict(row, policy='candidate')])))
    write(); report = audit(root)
    result = report['pairs'][0]['baseline']
    assert sum(p['frames'] for p in result['phases']) == 1500
    np.testing.assert_allclose(result['integrated'], [0, 60])
    np.testing.assert_allclose(sum(result['discounted']['0.99']['0']), .04*(1-.99**1500)/(1-.99))
    corrupt = scalar.copy(); corrupt[553] += .2
    shifted = inputs.copy(); shifted[50, 0, 3] = .05
    revived = motion.copy(); revived[500, :, -1] = 0
    for kwargs in [dict(s=corrupt), dict(i=shifted), dict(m=revived), dict(r=rates[:-1])]:
        write(**kwargs)
        try:
            audit(root)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f'Corruption was accepted: {kwargs.keys()}')
print('Full partition, discount sum, transition corruption, command timing and failure exclusion passed')
