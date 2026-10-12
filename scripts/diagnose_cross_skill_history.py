"""Frozen-sensor recurrence is a counterfactual probe, not physical replay."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
root=Path(__file__).resolve().parents[2]
artifact=root/'microduck-rl/artifacts/process-repair-v2-20261010'
trace=json.loads((root/'microduck-observer/docs/实测记录/附件/平台/2026-10-10/sit-trace/session.json').read_text(encoding='utf-8'))
rows=[r for r in trace['frames'] if r['type']=='inference' and r['kind']=='sitstand_sit']
selected=np.linspace(0,len(rows)-1,9,dtype=int)
options=ort.SessionOptions();options.intra_op_num_threads=1
getup=ort.InferenceSession(str(root/'microduck-observer/models/xgoduck_getup.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
results=[]
for index in selected:
    recorded=np.array(rows[index]['observation'],dtype=np.float32)
    for mode in ('zero','recorded','position'):
        obs=recorded.copy();obs[48:]=0
        obs[34:48]=np.zeros(14) if mode=='zero' else recorded[6:20] if mode=='position' else recorded[34:48]
        actions=[]
        for _ in range(150):
            value=getup.run(None,{'obs':obs[None]})[0][0]
            assert np.isfinite(value).all()
            actions.append(value.copy());obs[34:48]=value
        actions=np.asarray(actions)
        delta=np.diff(actions[-21:],axis=0)
        results.append({'recordedFrameIndex':int(index),'historyInitialization':mode,
            'last20RmsActionChangeRad':float(np.sqrt(np.mean(delta**2))),
            'last20MaxKneeFrameChangeDeg':np.degrees(np.max(np.abs(delta[:,[3,12]]),axis=0)).tolist(),
            'lastActionRad':actions[-1].tolist(),
            'initialRecordedSitPositionDeltaRad':recorded[6:20].tolist()})
output={'scope':'Getup tested counterfactually at nine recorded SIT observations with all commands zero; sensors held fixed, only raw action history recursively updated. Not an actual getup trace or closed-loop motion proof.',
        'hardwareAccess':False,'results':results}
(artifact/'cross-skill-history.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'cases':len(results),'maximumLast20RmsRad':max(v['last20RmsActionChangeRad'] for v in results),
    'byInitialization':{mode:max(v['last20RmsActionChangeRad'] for v in results if v['historyInitialization']==mode) for mode in ('zero','recorded','position')}}))
