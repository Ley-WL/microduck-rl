"""Collect privileged feedback demonstrations; split by whole initial states."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--speeds',type=float,nargs='+',default=[-.2,-.1,0.,.1,.2])
p.add_argument('--seeds',type=int,nargs='+',default=[207101,207107,207111])
p.add_argument('--feedback-vx-limit',type=float,default=.12)
p.add_argument('--feedback-query-vx-limit',type=float,default=.3)
a=p.parse_args();assert len(a.seeds)>=2 and len(set(a.seeds))==len(a.seeds)
assert len(set(a.speeds))==len(a.speeds)
root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
seeds=a.seeds;inputs=[];targets=[];validation=[];worlds=[]
for yaw in [-.6,-.3,0.,.3,.6]:
    folder=a.output/f'yaw{yaw:g}'
    subprocess.run([sys.executable,'-X','utf8',str(root/'scripts/evaluate_walk_runtime_cpu.py'),
        '--onnx',str(a.onnx),'--physics-profile','kissqy','--output',str(folder),
        '--speeds',*[str(v) for v in a.speeds],'--seeds',*[str(s) for s in seeds],
        '--trials','1','--seconds','30','--transport-ms','0','--turn-rate',str(yaw),
        '--feedback-teacher','--feedback-vx-limit',str(a.feedback_vx_limit),'--feedback-query-vx-limit',str(a.feedback_query_vx_limit)],check=True)
    for path in sorted(folder.glob('seed*.json')):
        frames=json.loads(path.read_text())['frames']
        valid=len(frames)==1500 and max(f['tilt'] for f in frames)<45.
        heldout=path.name.startswith(f'seed{seeds[-1]}-')
        worlds.append(dict(source=str(path.relative_to(a.output)),valid=valid,validation=heldout))
        if not valid:continue
        x=np.asarray([f['obs'] for f in frames],dtype=np.float32)
        y=np.asarray([f['raw'] for f in frames],dtype=np.float32)
        assert x.shape==(1500,61) and y.shape==(1500,14)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        inputs.append(x);targets.append(y);validation.append(np.full(len(x),heldout))
    (a.output/'progress.json').write_text(json.dumps(dict(completedYaw=yaw,worlds=len(worlds))))
assert len(worlds)==5*len(a.speeds)*len(seeds) and all(w['valid'] for w in worlds),'Feedback demonstration failed; inspect before training'
np.savez_compressed(a.output/'feedback-data.npz',observation=np.concatenate(inputs),target=np.concatenate(targets),validation=np.concatenate(validation))
audit=dict(worlds=worlds,seeds=seeds,frames=1500*len(worlds),trainingWorlds=sum(not w['validation'] for w in worlds),validationWorlds=sum(w['validation'] for w in worlds),
    speeds=a.speeds,feedbackVxLimit=a.feedback_vx_limit,feedbackQueryVxLimit=a.feedback_query_vx_limit,
    teacherPolicySha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest(),
    teacherSourceSha256=hashlib.sha256((root/'scripts/walk_feedback_teacher.py').read_bytes()).hexdigest(),
    dataSha256=hashlib.sha256((a.output/'feedback-data.npz').read_bytes()).hexdigest(),
    desiredCommandRetained=True,privilegedTeacherOnly=True,hardwareAccess=False)
(a.output/'data-audit.json').write_text(json.dumps(audit,indent=2))
print('feedback data complete',audit['frames'],audit['dataSha256'])
