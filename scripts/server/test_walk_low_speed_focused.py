"""Ensure the comparison changes only completed-stride reward weight."""
from mjlab_microduck.tasks import walk_low_speed_focused as stride,walk_low_speed_focused_control as control
from mjlab_microduck.tasks import walk_low_speed_stride as previous

a,b,old=control.make_cfg(),stride.make_cfg(),previous.make_cfg()
for field in ('sim','decimation','actions','observations','commands','events','terminations'):
    assert getattr(a,field)==getattr(b,field),field
assert a.scene.entities['robot'].articulation==b.scene.entities['robot'].articulation
assert b.scene.entities['robot'].articulation==old.scene.entities['robot'].articulation
for name in a.rewards:
    if name!='long_stride':assert a.rewards[name]==b.rewards[name],name
assert a.rewards['long_stride'].params==b.rewards['long_stride'].params
assert a.rewards['long_stride'].weight==1. and b.rewards['long_stride'].weight==6.
assert b.rewards['long_stride'].params['min_stride']==.06
assert b.rewards['long_stride'].params['max_air_credit']==1.
assert b.rewards['long_stride'].params['require_alternation']
assert b.rewards['long_stride'].params['complete_swing']
print('Focused command control/reward-only contrast/unchanged actuator contract passed')
