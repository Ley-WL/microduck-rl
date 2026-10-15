"""Compare vectorized teacher against independent per-world NumPy state."""
import numpy as np
import torch
from walk_feedback_teacher import FeedbackTeacher,TensorFeedbackTeacher

rng=np.random.default_rng(210101)
scalar=[FeedbackTeacher() for _ in range(8)];tensor=TensorFeedbackTeacher()
worst=0.
for step in range(700):
    command=np.zeros((8,3),dtype=np.float32)
    command[:,0]=np.linspace(-.2,.2,8)*(-1 if step>=350 else 1)
    command[:,2]=np.linspace(-.6,.6,8)
    enabled=np.full(8,step%150<100);enabled[0]=False
    actual=rng.normal(0,.3,(8,2)).astype(np.float32)
    before=command.copy()
    expected=np.stack([t.query(c,v,enabled=bool(e)) for t,c,v,e in zip(scalar,command,actual,enabled)])
    actual_tensor=tensor.query(torch.tensor(command),torch.tensor(actual),enabled=torch.tensor(enabled)).numpy()
    np.testing.assert_allclose(actual_tensor,expected,atol=2e-6,rtol=2e-6)
    np.testing.assert_array_equal(command,before)
    worst=max(worst,float(abs(actual_tensor-expected).max()))
print('Tensor feedback parity passed',worst)
