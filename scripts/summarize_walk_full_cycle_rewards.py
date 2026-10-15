"""Group paired full-cycle evidence without pooling speeds, yaw, or phases."""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('audit', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
data = json.loads(a.audit.read_text())
assert data['passed']
groups = []
for command in sorted({tuple(row['command']) for row in data['pairs']}):
    rows = [r for r in data['pairs'] if tuple(r['command']) == command]
    group = dict(command=command, paired=len(rows), phases=[])
    for phase in range(9):
        entry = dict(name=rows[0]['baseline']['phases'][phase]['name'])
        for label in ('baseline', 'candidate'):
            values = [r[label]['phases'][phase] for r in rows]
            entry[label] = dict(actual=np.mean([v['meanActual'] for v in values], axis=0).tolist(),
                meanTotalRate=float(np.mean([sum(v['meanRate']) for v in values])),
                totalIntegral=float(np.mean([sum(v['integrated']) for v in values])))
        delta = np.mean([np.asarray(r['candidate']['phases'][phase]['meanRate'])-
            r['baseline']['phases'][phase]['meanRate'] for r in rows], axis=0)
        entry['largestRateChanges'] = sorted(zip(data['terms'], delta.tolist()), key=lambda v: -abs(v[1]))[:8]
        entry['candidateHigherScoreWorlds'] = sum(sum(r['candidate']['phases'][phase]['meanRate']) >
            sum(r['baseline']['phases'][phase]['meanRate']) for r in rows)
        group['phases'].append(entry)
    for label in ('baseline', 'candidate'):
        group[label] = dict(integrated=float(np.mean([sum(r[label]['integrated']) for r in rows])),
            discounted={g:{start:float(np.mean([sum(r[label]['discounted'][g][start]) for r in rows]))
                for start in ['0','50','800']} for g in ['0.99','0.997']})
    groups.append(group)
a.output.write_text(json.dumps(dict(accepted=False, simulationOnly=True, groups=groups,
    failed=data['failed'], checks=data['checks']), indent=2), encoding='utf-8')
for group in groups:
    if group['command'][1] == 0:
        print('command', group['command'], 'full integral',
              group['baseline']['integrated'], group['candidate']['integrated'])
        for phase in group['phases']:
            print(phase['name'], 'vx', phase['baseline']['actual'][0], phase['candidate']['actual'][0],
                  'score', phase['baseline']['meanTotalRate'], phase['candidate']['meanTotalRate'],
                  'top changes', phase['largestRateChanges'][:3])
