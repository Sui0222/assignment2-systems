import torch
from torch import nn

x=torch.randn((4,512,2560),requires_grad=True) #4*512*2560*4/1000^2=20MiB

class RMSNorm(nn.Module):
    def __init__(self, hidden_size:int,eps:float=1e-5):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps

    def forward(self,x):
        rms=torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        x=x*rms
        return self.weight*x

def pack_hook(tensor):
    print("Saving residual:",f"shape={tensor.shape},",f"dtype={tensor.dtype},",f"grad_fn={tensor.grad_fn}",)
    return tensor

def unpack_hook(tensor):
    print("Loading residual:",f"shape={tensor.shape},",f"dtype={tensor.dtype},",f"grad_fn={tensor.grad_fn}",)
    return tensor

norm = torch.compile(RMSNorm(x.shape[-1]))

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = norm(x)
    y.sum().backward()
