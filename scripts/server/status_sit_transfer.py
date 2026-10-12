"""Read bounded status without changing a running training job."""
import json,os,re
from pathlib import Path
root=Path(__file__).resolve().parents[2];evidence=root/'evidence/train'
launch=json.loads((evidence/'launch.json').read_text())
log=(evidence/'train.log').read_text(errors='replace') if (evidence/'train.log').exists() else ''
text=re.sub(r'\x1b\[[0-9;]*m','',log)
iterations=re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)',text)
try:os.kill(launch['pid'],0);alive=True
except ProcessLookupError:alive=False
fields={}
for label in ('Mean sit_retention loss','Mean sit_onpolicy_retention loss','Mean sit_real_temporal loss',
              'Episode_Termination/nan_state','Curriculum/action_rate_weight','Time elapsed','ETA'):
    values=re.findall(re.escape(label)+r':\s*([^\n]+)',text)
    if values:fields[label]=values[-1].strip()
result={'launcherAlive':alive,'iteration':int(iterations[-1][0])+1 if iterations else 0,'budget':1000,
        'trainReportedDone':bool(iterations and int(iterations[-1][0])==999),'metrics':fields,
        'jobExit':(evidence/'job.exit').read_text().strip() if (evidence/'job.exit').exists() else None,
        'completed':(evidence/'completed.txt').exists(),
        'comparison':json.loads((evidence/'comparison.json').read_text()) if (evidence/'comparison.json').exists() else None}
print(json.dumps(result))
