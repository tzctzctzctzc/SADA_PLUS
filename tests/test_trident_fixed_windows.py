import pytest
import torch
from tools.trident_windows import fixed_windows,strided_windows


@pytest.mark.parametrize('left',[False,True])
def test_same_forward_and_mathematical_adjoint(left):
    torch.manual_seed(5)
    x=torch.randn(2,3,8,dtype=torch.double,requires_grad=True);other=x.detach().clone().requires_grad_()
    upstream=torch.randn(2,3,8,5,dtype=torch.double)
    a,b=strided_windows(x,4,left),fixed_windows(other,4,left)
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a,x,upstream)[0];gb=torch.autograd.grad(b,other,upstream)[0]
    torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda z:fixed_windows(z,4,left),(other,),eps=1e-6,atol=1e-5,rtol=1e-4)
