"""Early falls must remain in their intended command group and failed-pair list."""
import json
from pathlib import Path
import tempfile

from audit_walk_cpu_pair import audit

with tempfile.TemporaryDirectory() as directory:
    roots = [Path(directory) / name for name in ('baseline', 'candidate')]
    name = 'cpu-yaw0.6/seed1-trial0-speed0.035.json'
    frame = dict(qpos=[0., 0., 1.], velocity=[0., 0., 0.], command=[0., 0., 0.])
    for index, root in enumerate(roots):
        (root / name).parent.mkdir(parents=True)
        (root / 'job.exit').write_text('0')
        (root / name).write_text(json.dumps(dict(frames=[frame] * (2 if index == 0 else 1))))
        row = dict(source=name, completed=index == 0, meanCommand=[.035, .6],
                   meanActual=[.035, .6] if index == 0 else [float('nan')] * 2,
                   rolling1sMAE=[0., 0.], stepDistance=.05 if index == 0 else None,
                   stopMeanHorizontalSpeed=0.)
        (root / 'cpu-body-tracking.json').write_text(json.dumps(dict(results=[row])))
    result = audit(*roots)
    assert result['lost'] == [name]
    assert result['groups'][0]['command'] == [.035, .6]
    assert result['groups'][0]['candidate']['survived'] == 0
    assert result['groups'][0]['candidate']['actual'] == [None, None]
    json.dumps(result, allow_nan=False)
    frame['velocity'][0] = .1
    (roots[1] / name).write_text(json.dumps(dict(frames=[frame])))
    try:
        audit(*roots)
    except AssertionError:
        pass
    else:
        raise AssertionError('Different initial velocity must reject pairing')
print('Early failure retention, intended-command grouping, missing metrics and initial-state rejection passed')
