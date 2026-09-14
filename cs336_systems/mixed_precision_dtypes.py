import argparse

import torch
import torch.nn as nn


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x=self.fc1(x)
        print("fc1 output:",x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print("layer norm output:",x.dtype)
        x = self.fc2(x)
        return x

device="cuda"

model=ToyModel(in_features=8,out_features=4).to(device)
inputs=torch.randn(2,8,device=device)
targets=torch.randint(0,4,(2,),device=device)

with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
    print("parameter:",model.fc1.weight.dtype)
    logits=model(inputs)
    print("logits:",logits)
    loss=nn.functional.cross_entropy(logits,targets)
    print("loss:",loss)
    print("loss.dtype:",loss.dtype)
    print("logits.dtype:",logits.dtype)

loss.backward()
print("gradient:",model.fc1.weight.grad.dtype)






