import numpy as np
from walk_feedback_teacher import FeedbackTeacher

t=FeedbackTeacher()
for _ in range(500):q=t.query([.2,0,.6],[.1,.2])
assert .2<q[0]<=.3 and .6<q[2]<=1.2
assert np.max(np.abs(t.integral))<=.6
np.testing.assert_array_equal(t.query([0,0,0],[.4,1.]),np.zeros(3))
assert not t.integral.any() and not t.correction.any()
np.testing.assert_allclose(t.query([.16,0,.48],[.2,.6],enabled=False),[.16,0,.48])
assert not t.integral.any() and not t.correction.any()
q=t.query([.2,0,.6],[.4,1.])
assert np.abs(q[0]-.2)<=.00601 and np.abs(q[2]-.6)<=.02401
try:t.query([0,0,0],[float('nan'),0])
except ValueError:pass
else:raise AssertionError('Non-finite measurement accepted')
print('feedback teacher saturation/reset/slew/finite checks passed')

# Wider bounds only affect saturation; default behavior remains parity-tested.
wide=FeedbackTeacher(vx_limit=.25)
for _ in range(500):q=wide.query([-.05,0,0],[0,0])
assert q[0]<-.2 and q[0]>=-.3
np.testing.assert_allclose(wide.query([-.04,0,0],[0,0],enabled=False),[-.04,0,0])
for bound in (0,-1,.31,float('nan')):
    try:FeedbackTeacher(vx_limit=bound)
    except ValueError:pass
    else:raise AssertionError('Invalid bound accepted')
