"""Compare minimum-stride and matched wide-command control without changing PD."""
from mjlab_microduck.tasks import walk_low_speed_stride as stride,walk_low_speed_control as control
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1]/'test_walk_stride_floor.py'))
a,b=control.make_cfg(),stride.make_cfg()
for field in ('sim','decimation','actions','observations','commands','events','terminations'):
    assert getattr(a,field)==getattr(b,field),field
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
assert a.scene.entities['robot'].articulation.actuators[0].kp_fw==5.
assert set(stride.SPEEDS)=={-.4,-.3,-.2,-.1,-.05,0.,.05,.1,.2,.3,.4}
for name in a.rewards:
    if name!='long_stride':assert a.rewards[name]==b.rewards[name],name
assert a.rewards['long_stride'].weight==b.rewards['long_stride'].weight
assert b.rewards['long_stride'].params['min_stride']==.06
assert b.rewards['long_stride'].params['max_air_credit']==1.
assert b.rewards['long_stride'].params['require_alternation']
assert b.rewards['long_stride'].params['complete_swing']
print('Matched coverage/reward-only difference/PD/control invariants passed')
