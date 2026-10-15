import torch
from walk_pulse_reference import PulseReference
teacher=PulseReference(4,'cpu');cmd=torch.tensor([[.05,0,0],[-.05,0,0],[0,0,0],[.1,0,0]]);saved=cmd.clone();rows=[]
for i in range(1000):rows.append(teacher.step(cmd,torch.full((4,),i*.02)))
assert torch.equal(cmd,saved)
x=torch.stack(rows);assert not x[:,2].any();assert not x[:,:,1:].any()
assert torch.allclose(x[-500:,0,0].mean(),torch.tensor(.04),atol=.001)
assert torch.allclose(x[-500:,1,0].mean(),torch.tensor(-.04),atol=.001)
assert torch.allclose(x[-500:,3,0].mean(),torch.tensor(.08),atol=.001)
before=teacher.query.clone();teacher.reset(torch.tensor([True,False,False,False]));assert not teacher.query[0].any();assert torch.equal(before[1:],teacher.query[1:])
print('Original commands unchanged, pulse mean/sign, zero command, partial reset passed')
