"""Summarize measured gate suppression and training regression; no retraining."""
import json,re
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];base=root.parent
out=base/'setup/v13-regression'
report={}
trace=json.loads((out/'v12-gates-trace.json').read_text())
active=[x for x in trace if 4<=x['time']<7 and not x['ended']]
quality=np.array([x['straightQuality'] for x in active])
report['sourceGaitUnderV13InstantStraightGate']={
    'meanQuality':float(quality.mean()),'medianQuality':float(np.median(quality)),
    'fractionBelowOnePercent':float((quality<.01).mean()),
    'meanAbsYawRate':float(np.mean([abs(x['yawRate']) for x in active])),
    'meanAbsLateralVelocity':float(np.mean([abs(x['velocity'][1]) for x in active])),
    'scope':'first world, t=4..7s, same frozen v12 ONNX, .1m/s, 8-world run'}
rewards={}
for x in active:
    for name,value in x['weightedRewardTerms'].items():rewards[name]=rewards.get(name,0.)+value
report['sourceGaitWeightedRewardsDuring3Seconds']=rewards
log=(base/'runs/hd1910-walk-v13-slow1000.log').read_text(errors='replace')
log=re.sub(r'\x1b\[[0-9;]*m','',log)
snapshots=[]
for block in re.split(r'Learning iteration ',log)[1:]:
    n=int(block.split('/')[0])
    if n not in (0,100,250,300,500,700,750,999):continue
    fields={name:float(value) for name,value in re.findall(
        r'(Mean reward|Mean episode length|Episode_Reward/[\w]+):\s*([-+0-9.]+)',block)}
    snapshots.append({'iteration':n,**fields})
report['trainingSnapshots']=snapshots
report['checkpointComparisons']=json.loads((out/'summary.json').read_text())
parity=out/'v13-parity.json'
if parity.exists():report['checkpointOnnxMaxAbsError']=json.loads(parity.read_text())['checkpointOnnxMaxAbsError']
(out/'diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['sourceGaitUnderV13InstantStraightGate']))
for x in snapshots:
    print('iter',x['iteration'],'episodeSteps',x.get('Mean episode length'),
        'strideReward',x.get('Episode_Reward/stride_quality'),
        'trackingReward',x.get('Episode_Reward/track_linear_velocity'))
