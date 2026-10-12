"""Visualize same-start nominal vs diagnostic physical-pitch stress."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'artifacts/sit-transfer-train-20261010'
def load(tag):
    report=json.loads((root/('reset-paired-'+tag)/'summary.json').read_text(encoding='utf-8'))
    return {(g['scenario'],g['seed'],r['trial']):r for g in report['groups'] for r in g['results']}
original=load('nominal');stress=load('stress')
assert set(original)==set(stress)
for key in original:np.testing.assert_allclose(original[key]['initialQpos'],stress[key]['initialQpos'],atol=1e-12)
failures=[key for key in original if original[key]['success'] and not stress[key]['success']]
case=failures[0]
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for tag,color in (('nominal','#287ca6'),('stress','#c44742')):
    path=root/('reset-paired-'+tag)/('-'.join(map(str,case))+'.policy.json')
    rows=json.loads(path.read_text(encoding='utf-8'))
    rows=[r for r in rows if 1.8<=r['time']<5]
    time=np.array([r['time'] for r in rows])
    axes[0].plot(time,np.degrees([r['target'][3] for r in rows]),label=tag,color=color)
    axes[1].plot(time,[r['observation'][1] for r in rows],color=color)
    trace=json.loads(path.with_suffix('').with_suffix('.json').read_text(encoding='utf-8'))['trace']
    trace=[r for r in trace if 1.8<=r['time']<5]
    axes[2].plot([r['time'] for r in trace],[r['tiltDeg'] for r in trace],color=color)
axes[0].set_ylabel('Left knee target (deg)');axes[0].legend()
axes[1].set_ylabel('Observed pitch gyro (rad/s)')
axes[2].set_ylabel('Body tilt (deg)');axes[2].set_xlabel('Time (s)')
for ax in axes:
    ax.axvspan(2,2.8,color='#888888',alpha=.12);ax.grid(alpha=.2)
fig.suptitle('Added transition validation: same initial pose, physical pitch impulses\nDiagnostic stress hypothesis; no robot movement or hardware acceptance',fontsize=12)
fig.tight_layout();fig.savefig(root/'validation-comparison.png',dpi=160);plt.close(fig)
report={'nominalSuccesses':sum(r['success'] for r in original.values()),'stressSuccesses':sum(r['success'] for r in stress.values()),
        'trials':len(original),'initialStatesExactlyPaired':True,'lostSuccessfulCases':list(map(list,failures)),
        'plottedCase':list(case),'diagnosticStressHypothesis':True,'hardwareValidated':False}
(root/'validation-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
