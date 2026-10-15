"""Compare the retention penalty and gradients to torch's distribution KL."""
import torch
from walk_actor_retention import gaussian_mean_kl

torch.manual_seed(258901)
x=torch.randn(17,14,dtype=torch.float64,requires_grad=True)
y=torch.randn_like(x);sigma=torch.linspace(.002,.02,14,dtype=torch.float64,requires_grad=True)
actual=gaussian_mean_kl(x,y,sigma)
reference=torch.distributions.kl_divergence(torch.distributions.Normal(x,sigma.detach()),
    torch.distributions.Normal(y,sigma.detach())).sum(-1).mean()
torch.testing.assert_close(actual,reference)
gradient=torch.autograd.grad(actual,x,retain_graph=True)[0]
torch.testing.assert_close(gradient,torch.autograd.grad(reference,x)[0])
torch.testing.assert_close(gaussian_mean_kl(x*.1,y*.1,sigma*.1),actual)
assert gaussian_mean_kl(y,y,sigma)==0
actual.backward();assert sigma.grad is None
for bad in (torch.zeros_like(sigma),-torch.ones_like(sigma)):
    try:gaussian_mean_kl(x,y,bad)
    except AssertionError:pass
    else:raise AssertionError('Invalid noise scale accepted')
print('Distribution KL, exact gradients, unit scaling, zero anchor and detached noise passed')
