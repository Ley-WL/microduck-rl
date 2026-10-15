"""Two arms differ only in the long-stride reward, not hardware or runtime."""
import runpy
from pathlib import Path
from dataclasses import asdict
from mjlab_microduck.tasks import walk_tracking_transfer as control, walk_stride_transfer as stride
runpy.run_path(str(Path(__file__).with_name('test_walk_tracking_transfer.py')))
a,b=control.make_cfg(),stride.make_cfg()
assert a.sim==b.sim and a.decimation==b.decimation
assert a.actions==b.actions and a.observations==b.observations
assert a.commands==b.commands and a.events==b.events and a.terminations==b.terminations
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
for name in a.rewards:
    if name!='long_stride':assert a.rewards[name]==b.rewards[name],name
assert b.rewards['long_stride'].weight==6.
assert not b.rewards['long_stride'].params['complete_swing']
assert b.rewards['long_stride'].params['require_alternation']
assert asdict(control.rl_cfg().algorithm)==asdict(stride.rl_cfg().algorithm)
print('two-arm isolation and hardware/control invariants passed')
