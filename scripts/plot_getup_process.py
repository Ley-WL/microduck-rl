"""Plot recovery and settled knee commands separately; offline evidence only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'artifacts/process-repair-v2-20261010/getup-process'
case='inverted-9001-0'
fig,axes=plt.subplots(3,1,figsize=(10,8))
for condition,color in [('nominal','#267ca6'),('stress','#cf5d4d')]:
    frames=json.loads((root/condition/(case+'.policy.json')).read_text(encoding='utf-8'))
    trace=json.loads((root/condition/(case+'.json')).read_text(encoding='utf-8'))['trace']
    time=np.array([r['time'] for r in frames]);targets=np.degrees([r['target'][3] for r in frames])
    axes[0].plot(time,targets,color=color,label=condition)
    axes[1].plot([r['time'] for r in trace],[r['tiltDeg'] for r in trace],color=color)
    late=time>=12
    axes[2].plot(time[late],targets[late],color=color)
axes[0].set_ylabel('Left knee target (deg)');axes[0].legend()
axes[1].set_ylabel('Body tilt (deg)')
axes[2].set_ylabel('Settled left knee (deg)');axes[2].set_xlabel('Time (s)')
for ax in axes:ax.grid(alpha=.2)
axes[0].axvspan(1,1.8,alpha=.12,color='#888888');axes[1].axvspan(1,1.8,alpha=.12,color='#888888')
fig.suptitle('Getup: paired inverted-start recovery and final standing commands\n48/48 nominal and 48/48 diagnostic stress successes; no physical robot trial',fontsize=12)
fig.tight_layout();fig.savefig(root/'getup-process.png',dpi=150)
print(str(root/'getup-process.png'))
