"""Check proposed command-responsive bias without mistaking it for gait evidence."""
import numpy as np
from replay_walk_current_command_bias import current_command_bias
from audit_walk_full_cycle_rewards import relative_bias

x=np.zeros((1500,2,6));x[:,0,3]=.025;x[:,1,3]=-.025
new=current_command_bias(x,.02);old=relative_bias(x,.02)
np.testing.assert_allclose(new,-4.)
assert abs(old[0,0])<.001 and old[-1,0]<-3.9
np.testing.assert_array_equal(new[:,0],new[:,1])
# With zero commands both implementations have exactly the same recurrence.
x=np.random.default_rng(257901).normal(0,.1,(1500,3,6));x[:,:,3:]=0
np.testing.assert_allclose(current_command_bias(x,.02),relative_bias(x,.02),atol=1e-12)
# Physical memory decays at stop; no discontinuous deletion of actual movement.
x=np.zeros((300,1,6));x[:100,0,0]=.05;x[:100,0,3]=.05
r=current_command_bias(x,.02)
assert r[100,0]<0 and abs(r[-1,0])<abs(r[100,0])*.001
print('Immediate command deficit, sign symmetry, rest parity and stop-memory decay passed')
