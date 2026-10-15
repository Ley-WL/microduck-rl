"""Create one isolated fixed-delay case; refuse duplicate launches.

Run with the training virtualenv, using its installed delay implementation.
No GPU learning is launched here; the subsequent prepare launcher has capacity gates.
"""
from pathlib import Path
import hashlib
import json
import shutil
import dataclasses
import inspect
import sys
import torch
from mjlab.utils.buffers.delay_buffer import DelayBuffer
from mjlab.actuator.actuator import Actuator
from bam.actuator import VoltageControlledActuator

base = Path('/home/leiwu30/microduck-training')
source = base / 'walk-encoder-transfer-20261013'
target = base / 'walk-delay16-prepare-20261013'
resume = '--audit-existing' in sys.argv
assert resume or not target.exists(), 'Inspect existing case before recreating anything'
if resume:
    assert target.exists()
    assert not (target / 'evidence/prepare-launch.json').exists(), 'Follow existing prepare PID'
assert (source / 'evidence/job.exit').read_text().strip() == '0'
manifest = json.loads((source / 'evidence/launch.json').read_text())
for file, sha in manifest['sourceHashes'].items():
    assert hashlib.sha256((source / file).read_bytes()).hexdigest() == sha, file
if not resume:
    target.mkdir()
    for folder in ('src', 'scripts', 'inputs'):
        shutil.copytree(source / folder, target / folder)
    (target / 'evidence').mkdir()
here = Path(__file__).resolve().parent
if not resume:
    shutil.copy2(here / 'walk_delay_transfer.py', target / 'src/mjlab_microduck/tasks/walk_delay_transfer.py')
    for name in ('prepare_walk_surface_transfer.py', 'train_walk_surface_transfer.py', 'launch_walk_surface_prepare.py'):
        p = target / 'scripts/server' / name
        p.write_text(p.read_text().replace('from mjlab_microduck.tasks.walk_surface_transfer import',
                                          'from mjlab_microduck.tasks.walk_delay_transfer import'))
sys.path.insert(0, str(target / 'src'))
import os
os.environ['MICRODUCK_WALK_SURFACE_INPUTS'] = str(target / 'inputs')
from mjlab_microduck.tasks.walk_surface_transfer import make_cfg as original
from mjlab_microduck.tasks.walk_delay_transfer import make_cfg as changed
from mjlab_microduck.tasks.walk_surface_transfer import rl_cfg

def simple(v):
    if isinstance(v, dict): return {str(k):simple(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [simple(x) for x in v]
    if callable(v): return v.__module__ + '.' + v.__qualname__
    if hasattr(v, 'tolist'): return v.tolist()
    if v is None or isinstance(v, (str,int,float,bool)): return v
    return str(v)

for surface in (False, True):
    a = original(surface=surface)
    b = changed(surface=surface)
    acts = b.scene.entities['robot'].articulation.actuators
    assert all(x.delay_min_lag == x.delay_max_lag == 8 for x in acts)
    for act in acts: act.delay_min_lag = act.delay_max_lag = 7
    assert simple(dataclasses.asdict(a)) == simple(dataclasses.asdict(b))
assert rl_cfg().num_steps_per_env == 24

# Exercise the installed Actuator.apply_delay with exactly its real buffer.
# Warm history before a target step; timestamps are pre-integration physics ticks.
@dataclasses.dataclass
class Target:
    position_target: torch.Tensor

traces = {}
for lag in (7,8):
    class Holder:
        command_field = 'position'
    holder = Holder()
    holder._delay_buffer = DelayBuffer(lag,lag,batch_size=4,device='cpu')
    for _ in range(20):
        Actuator.apply_delay(holder, Target(torch.zeros(4,14)))
    outputs = []
    for tick in range(12):
        result = Actuator.apply_delay(holder, Target(torch.ones(4,14)))
        outputs.append(float(result.position_target[0,0]))
    first = next(i for i,x in enumerate(outputs) if x == 1)
    assert first == lag, (lag,outputs)
    traces[str(lag)] = dict(outputs=outputs,firstAppliedTick=first,firstAppliedMs=2*first)

# BAM's control/torque methods do not apply JSON command_delay on top of buffer.
methods = {name: inspect.getsource(getattr(VoltageControlledActuator,name))
           for name in ('compute_control','compute_torque')}
assert all('command_delay' not in code for code in methods.values())
out = dict(passed=5,onlyConfigDifference='fixed actuator lag 7 -> 8 physics ticks',
           trace=traces,optimizerSteps=0,hardwareAccess=False,
           measuredHardwareLatency=False,jsonDelayAddedByBamControl=False)
(target / 'evidence/delay-implementation-audit.json').write_text(json.dumps(out,indent=2))
for name, obj in [('actuator',Actuator),('delay_buffer',DelayBuffer),('voltage_actuator',VoltageControlledActuator)]:
    p = Path(inspect.getfile(obj))
    shutil.copy2(p, target / 'evidence' / (name + '-installed.py'))
(target / 'evidence/bam-control-methods.json').write_text(json.dumps(methods,indent=2))
print(json.dumps(dict(case=str(target),audit=out)))
