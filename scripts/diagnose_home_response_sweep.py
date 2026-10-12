"""Simulation sensitivity only; BAM kp is NOT mapped to HD1910 raw PID."""
from pathlib import Path
source=Path(__file__).with_name('diagnose_home_closed_loop.py').read_text(encoding='utf-8')
exec(compile(source.split('results=[]')[0],str(Path(__file__).with_name('diagnose_home_closed_loop.py')),'exec'))
results=[]
for gain in [10,20,32]:
 ns['bam'].actuator.kp=gain;ns['a'].reference_pose=False;ns['a'].official_targets=True
 ns['reset']();trace=[]
 for tick in range(1000):
  tilt=ns['step']();data=ns['data']
  trace.append(dict(time=float(data.time),tiltDeg=tilt,height=float(data.qpos[2]),position=data.qpos[ns['qadr']].tolist(),target=ns['last_target'].tolist()))
 tail=trace[-250:];q=np.array([x['position'] for x in tail]);target=np.array([x['target'] for x in tail])
 r=dict(bamGain=gain,seconds=20,maxTiltDeg=max(x['tiltDeg'] for x in trace),tailMaxTiltDeg=max(x['tiltDeg'] for x in tail),endHeight=trace[-1]['height'],maxPhysicalChangeDeg=float(np.max(np.abs(np.degrees(np.array([x['position'] for x in trace])-ns['home'])))),maxTailPositionSpanDeg=float(np.max(np.degrees(np.ptp(q,axis=0)))),maxTailTargetSpanDeg=float(np.max(np.degrees(np.ptp(target,axis=0)))))
 results.append(r);print(json.dumps(r),flush=True)
 out=root.parent/'microduck-observer/docs/实测记录/附件/HAT/2026-10-04/home-simulation'
 (out/('gain-'+str(gain)+'.json')).write_text(json.dumps(trace),encoding='utf-8')
(out/'response-sweep.json').write_text(json.dumps(dict(hardwareAccess=False,note='synthetic BAM gain sweep; no equivalence asserted to hardware kpRaw=32',results=results),indent=2),encoding='utf-8')
