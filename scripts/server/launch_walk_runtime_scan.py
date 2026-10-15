"""CPU-only frozen simulation; source locked, no deployment or model updates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
evidence=ROOT/'evidence';evidence.mkdir(exist_ok=True)
assert not (evidence/'launch.json').exists(), 'Track existing PID instead of launching again'
spec=importlib.util.spec_from_file_location('walk_scan',ROOT/'scripts/evaluate_walk_runtime_cpu.py')
scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)
cmd=scan.command_step(np.zeros(3,dtype=np.float32),np.asarray([-.4,0,.5],dtype=np.float32))
np.testing.assert_allclose(cmd,[-.08,0,.1],atol=1e-7,rtol=0)
first=scan.action_step(np.zeros(14,dtype=np.float32),np.ones(14,dtype=np.float32),False)
np.testing.assert_allclose(first,.9,atol=1e-7,rtol=0)
second=scan.action_step(first,-np.ones(14,dtype=np.float32),True)
np.testing.assert_allclose(second[5:9],0,atol=1e-7,rtol=0)
np.testing.assert_allclose(second[[0,1,2,3,4,9,10,11,12,13]],-.36,atol=1e-7,rtol=0)
np.testing.assert_allclose(scan.mechanical.encode_target(np.full(14,10)),
    scan.mechanical.encode_target(np.asarray(list(scan.mechanical.LIMITS.values()))[:,1]),atol=0,rtol=0)
observed=scan.observed_angles(np.full(14,1.3))
assert observed[8]>scan.mechanical.LIMITS['head_roll'][1], 'Feedback must not use target clipping'
assert np.max(np.abs(observed-1.3))<=scan.mechanical.ENCODER_STEP/2
np.testing.assert_allclose(scan.observed_angles(np.zeros(14)),0,atol=0,rtol=0)
from types import SimpleNamespace
class ActuatorProbe:
    def compute_control(self,target,q,dq,dt):
        np.testing.assert_array_equal(q,np.full(14,.03))
        np.testing.assert_array_equal(dq,np.full(14,2.))
        return np.zeros(14)
    def compute_torque(self,control,limited,q,dq):return np.zeros(14)
probe=scan.BacklashController.__new__(scan.BacklashController)
probe.qpos_indexes=np.arange(14);probe.passive_qadr=np.arange(14,28);probe.dof_indexes=np.arange(14);probe.act_indexes=np.arange(14)
probe.q_target=np.zeros(14);probe.last_ts=0
probe.mujoco_data=SimpleNamespace(qpos=np.r_[np.full(14,.02),np.full(14,.01)],qvel=np.r_[np.full(14,2.),np.full(14,100.)],time=.002,
    ctrl=np.zeros(14),qfrc_bias=np.zeros(28),qfrc_constraint=np.zeros(28),qfrc_actuator=np.zeros(28),
    efc_type=np.array([],dtype=int),efc_id=np.array([],dtype=int),efc_force=np.array([]))
probe.model=SimpleNamespace(actuator=ActuatorProbe(),compute_frictions=lambda *args:(np.zeros(14),np.zeros(14)))
probe.mujoco_model=SimpleNamespace(dof_frictionloss=np.zeros(28),dof_damping=np.zeros(28))
before=probe.mujoco_data.qpos.copy();probe.update();np.testing.assert_array_equal(before,probe.mujoco_data.qpos)
(evidence/'arithmetic-tests.json').write_text(json.dumps(dict(passed=8,hardwareAccess=False,
    references='Rust walk_profile first .9 / second head0 leg-.36, command -.08/0/.1',
    notFullDynamicsParity=True),indent=2))
sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ('scripts','src','inputs') for p in (ROOT/folder).rglob('*')
    if p.is_file() and '__pycache__' not in str(p)}
args=['scripts/evaluate_walk_runtime_cpu.py','--onnx','inputs/walk.onnx','--body-contract',
    'inputs/body-contract.json','--output','evidence/normal','--seconds','20']
commands=json.loads((ROOT/'inputs/commands.json').read_text()) if (ROOT/'inputs/commands.json').exists() else [args]
runner=ROOT/'run_scan.py'
runner.write_text('''import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
status=1
try:
    launch=json.loads((root/'evidence/launch.json').read_text())
    for arguments in launch['commands']:
        subprocess.run([sys.executable,*arguments],cwd=root,check=True)
    changed=[p for p,h in launch['sourceHashes'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    (root/'evidence/source-check.json').write_text(json.dumps({'passed':True,'files':len(launch['sourceHashes'])}))
    status=0
finally:
    (root/'evidence/job.exit').write_text(str(status))
    with tarfile.open(root/'complete.tar.gz','w:gz') as archive:
        for name in ('src','scripts','inputs','evidence','run_scan.py'):archive.add(root/name,arcname=name)
    (root/'complete.sha256').write_text(hashlib.sha256((root/'complete.tar.gz').read_bytes()).hexdigest())
sys.exit(status)
''')
# Write manifest before spawning; runner reads it only after this file exists.
manifest=dict(training=False,hardwareAccess=False,device='cpu',commands=commands,sourceHashes=sources)
(evidence/'launch.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='1'
with (evidence/'job.log').open('wb') as stream:
    process=subprocess.Popen([os.sys.executable,str(runner)],cwd=ROOT,env=env,
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
manifest['pid']=process.pid;(evidence/'pid').write_text(str(process.pid))
print(json.dumps({k:v for k,v in manifest.items() if k!='sourceHashes'}))
