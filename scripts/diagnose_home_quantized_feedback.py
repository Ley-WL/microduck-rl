"""CPU rehearsal with HD1910 encoder quantization and previous-frame finite differences."""
from pathlib import Path
source=Path(__file__).with_name('diagnose_home_closed_loop.py').read_text(encoding='utf-8')
exec(compile(source.split('results=[]')[0],str(Path(__file__).with_name('diagnose_home_closed_loop.py')),'exec'))
feedback=json.loads((root.parent/'microduck-observer/docs/实测记录/附件/HAT/2026-10-04/model-home-supported/shadow.json').read_text(encoding='utf-8'))
refs=np.array([feedback['before']['calibration']['joints']['references'][str(i)] for i in [20,21,22,23,24,30,31,32,33,10,11,12,13,14]])
ns['encoder_refs']=refs
step_source=preview.read_text(encoding='utf-8').split('def step():')[1].split('\nreset()\nprint(')[0]
step_source='def step():'+step_source
step_source=step_source.replace('global last_action,last_vel','global last_action,last_vel,last_target,last_obs,previous_q')
step_source=step_source.replace('    obs=np.concatenate', '    observed_q=(encoder_refs-np.rint(encoder_refs-data.qpos[qadr]*4096/(2*np.pi)))*2*np.pi/4096\n    obs=np.concatenate')
step_source=step_source.replace('data.qpos[qadr]-home','observed_q-home')
step_source=step_source.replace('    last_vel=data.qvel[vadr].copy()', '    last_obs=obs.copy()\n    last_vel=(observed_q-previous_q)/.02\n    previous_q=observed_q.copy()')
step_source=step_source.replace('    for _ in range(10):\n        history.append', '    last_target=target.copy()\n    for _ in range(10):\n        history.append')
exec(compile(step_source,str(preview),'exec'),ns)
ns['a'].reference_pose=False;ns['a'].official_targets=True;ns['reset']()
ns['previous_q']=(refs-np.rint(refs-ns['home']*4096/(2*np.pi)))*2*np.pi/4096
trace=[]
for tick in range(1000):
 tilt=ns['step']();data=ns['data'];trace.append(dict(time=float(data.time),tiltDeg=tilt,height=float(data.qpos[2]),position=data.qpos[ns['qadr']].tolist(),target=ns['last_target'].tolist()))
tail=trace[-250:];q=np.array([x['position'] for x in tail]);target=np.array([x['target'] for x in tail])
result=dict(hardwareAccess=False,seconds=20,maxTiltDeg=max(x['tiltDeg'] for x in trace),tailMaxTiltDeg=max(x['tiltDeg'] for x in tail),endHeight=trace[-1]['height'],maxTailPositionSpanDeg=float(np.max(np.degrees(np.ptp(q,axis=0)))),maxTailTargetSpanDeg=float(np.max(np.degrees(np.ptp(target,axis=0)))))
out=root.parent/'microduck-observer/docs/实测记录/附件/HAT/2026-10-04/home-simulation'
(out/'quantized-feedback.json').write_text(json.dumps(dict(summary=result,trace=trace)),encoding='utf-8');print(json.dumps(result))
