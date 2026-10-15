"""Reject mismatched surviving commands while retaining reset failure evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

phases=['startupForward','steadyForward','stopFirst','startupReverse','steadyReverse','stopSecond']
audit=dict(terms=['tracking'],phases=phases,samples=[[300]*6]*2,
    meanCommand=[[[.2,0,0]]*6]*2,meanActual=[[[.2,0,0]]*6]*2,
    meanWeightedRate=[[[1.]]*6]*2,meanLegacyWeightedRate=[[[1.]]*6]*2)
b=dict(seed=1,policy='baseline',survival=[False,True],initialStateSha256='same',
    commandTraceSha256='reset',commandWorldSha256=['reset-world','same-world'],rewardAudit=audit)
c=copy.deepcopy(b);c.update(policy='candidate',survival=[True,True],commandTraceSha256='no-reset',
    commandWorldSha256=['other-world','same-world'])
c['rewardAudit']['meanCommand'][0]=[[99.,0,0]]*6
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);source=root/'input.json';output=root/'report.json'
    def run(candidate):
        source.write_text(json.dumps(dict(results=[b,candidate],comparisons=[dict(seed=1)])))
        return subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'analyze_walk_yaw_reward_audit.py'),
            str(source),'--output',str(output)],capture_output=True,text=True)
    result=run(c);assert result.returncode==0,result.stderr
    report=json.loads(output.read_text());assert len(report['pairs'])==2 and len(report['excludedWorlds'])==1
    mismatch=copy.deepcopy(c);mismatch['commandWorldSha256'][1]='different-surviving-command'
    assert run(mismatch).returncode!=0,'Never accept mismatched commands for a surviving world'
    missing=copy.deepcopy(c);del missing['commandWorldSha256']
    assert run(missing).returncode!=0,'Batch mismatch needs per-world evidence'
print('Reset failure retained; surviving mismatch and missing evidence correctly rejected')
