"""Compare complete evaluation contracts before attributing model regressions."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
baseline=json.loads((root/'weight-1p0.metadata.json').read_text())
paths=list((root/'evidence').glob('**/train.metadata.json'))+list((root/'evidence').glob('**/policy.metadata.json'))
keys=['jointOrder','homeRadians','actuatorParamsSha256','simulation','identifiedCommandDelaySeconds',
      'actionEmaOldWeight','actionEmaNewWeight','lastActionObservation','nominalTrunkHeightM',
      'nominalSeatedTrunkHeightM','sitJointOverrides','trainingTargetClipRadians','robotSourceHashes','skillPhysicsRevision']
output=[]
for path in paths:
    meta=json.loads(path.read_text())
    differences={key:{'baseline':baseline.get(key),'candidate':meta.get(key)} for key in keys if baseline.get(key)!=meta.get(key)}
    output.append({'path':str(path.relative_to(root)),'differences':differences})
print(json.dumps(output))
