"""Offline candidates inspired by xgoduck runtime; no hardware PID mapping assumed."""
from pathlib import Path
source=Path(__file__).with_name('diagnose_home_closed_loop.py').read_text(encoding='utf-8')
exec(compile(source.split('results=[]')[0],str(Path(__file__).with_name('diagnose_home_closed_loop.py')),'exec'))
ns['filtered_action']=np.zeros(14,dtype=np.float32);ns['action_alpha']=0.
step_source=preview.read_text(encoding='utf-8').split('def step():')[1].split('\nreset()\nprint(')[0]
step_source='def step():'+step_source
step_source=step_source.replace('global last_action,last_vel','global last_action,last_vel,last_target,filtered_action')
step_source=step_source.replace('    target=home+action','    filtered_action=action_alpha*filtered_action+(1-action_alpha)*action\n    target=home+filtered_action')
step_source=step_source.replace('    for _ in range(10):\n        history.append','    last_target=target.copy()\n    for _ in range(10):\n        history.append')
exec(compile(step_source,str(preview),'exec'),ns)
results=[]
for gain,alpha in [(6,0.),(6,.45),(32,.45)]:
 ns['bam'].actuator.kp=gain;ns['a'].reference_pose=False;ns['a'].official_targets=True;ns['action_alpha']=alpha
 ns['reset']();ns['filtered_action']=np.zeros(14,dtype=np.float32);trace=[]
 for tick in range(1000):
  tilt=ns['step']();data=ns['data'];trace.append(dict(time=float(data.time),tiltDeg=tilt,height=float(data.qpos[2]),position=data.qpos[ns['qadr']].tolist(),target=ns['last_target'].tolist()))
 tail=trace[-250:];q=np.array([x['position'] for x in tail]);target=np.array([x['target'] for x in tail]);r=dict(bamGain=gain,actionAlpha=alpha,seconds=20,maxTiltDeg=max(x['tiltDeg'] for x in trace),tailMaxTiltDeg=max(x['tiltDeg'] for x in tail),endHeight=trace[-1]['height'],maxTailPositionSpanDeg=float(np.max(np.degrees(np.ptp(q,axis=0)))),maxTailTargetSpanDeg=float(np.max(np.degrees(np.ptp(target,axis=0)))))
 results.append(r);print(json.dumps(r),flush=True)
 out=root.parent/'microduck-observer/docs/实测记录/附件/HAT/2026-10-04/xgoduck-review'
 (out/('simulation-gain-'+str(gain)+'-alpha-'+str(alpha)+'.json')).write_text(json.dumps(trace),encoding='utf-8')
(out/'simulation-profile-summary.json').write_text(json.dumps(dict(hardwareAccess=False,modelSha256=hashlib.sha256(policy.read_bytes()).hexdigest(),note='Same current v5/replica simulation, only synthetic BAM gain and action EMA varied. No firmware D model or raw PID mapping asserted.',results=results),indent=2),encoding='utf-8')
