"""Compare completed CPU screens without dropping failures or mixing directions."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def finite_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def audit(baseline, candidate):
    rows = []
    summaries = []
    for root in (baseline, candidate):
        assert (root / 'job.exit').read_text().strip() == '0'
        data = json.loads((root / 'cpu-body-tracking.json').read_text())
        summaries.append({r['source'].replace('\\', '/'): r for r in data['results']})
    assert summaries[0] and summaries[0].keys() == summaries[1].keys()
    for name in sorted(summaries[0]):
        a, b = [json.loads((root / name).read_text())['frames'] for root in (baseline, candidate)]
        assert a[0]['qpos'] == b[0]['qpos'], name
        assert a[0]['velocity'] == b[0]['velocity'], name
        length = min(len(a), len(b))
        assert [f['command'] for f in a[:length]] == [f['command'] for f in b[:length]], name
        match = re.fullmatch(r'cpu-yaw([^/]+)/seed\d+-trial\d+-speed(.+)\.json', name)
        assert match, name
        rows.append(dict(source=name, command=[float(match[2]), float(match[1])],
                         baseline=summaries[0][name], candidate=summaries[1][name],
                         commonCommandFrames=length))
    groups = []
    # Group by commanded speed AND yaw, preserving backward/straight/turn cases.
    commands = sorted({tuple(r['command']) for r in rows})
    for command in commands:
        selected = [r for r in rows if tuple(r['command']) == command]
        group = dict(command=list(command), worlds=len(selected))
        for label in ('baseline', 'candidate'):
            data = [r[label] for r in selected]
            group[label] = dict(survived=sum(r['completed'] for r in data),
                actual=np.mean([r['meanActual'] for r in data], axis=0).tolist(),
                rolling1sMAE=np.mean([r['rolling1sMAE'] for r in data], axis=0).tolist(),
                footTravelIncludingNoSteps=float(np.mean([r['stepDistance'] or 0. for r in data])),
                stopSpeed=float(np.mean([r['stopMeanHorizontalSpeed'] for r in data])))
        groups.append(group)
    return finite_json(dict(simulationOnly=True, accepted=False, paired=len(rows),
                lost=[r['source'] for r in rows if r['baseline']['completed'] and not r['candidate']['completed']],
                groups=groups, rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('paired', 'lost', 'groups')}, indent=2))
