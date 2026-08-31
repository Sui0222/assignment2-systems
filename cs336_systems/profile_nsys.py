import argparse
import gc
import torch
from torch.cuda import nvtx

from config import CONFIGS
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", choices=CONFIGS.keys(), default="small")
parser.add_argument("--mode", choices=["fwd", "fwd_bwd", "full"], default="full")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--context_length", type=int, default=None, help="Override default context length")
parser.add_argument("--warmup_steps", type=int, default=3)
args = parser.parse_args()

config = CONFIGS[args.model_size]
seq_len = args.context_length if args.context_length is not None else config.context_length

# 初始化 Model 與 Optimizer
model = (
    BasicsTransformerLM(
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        context_length=seq_len,
    )
    .cuda()
    .train()
)

optimizer = AdamW(model.parameters(), lr=1e-3)

# 產生 Dummy Data
inputs = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device="cuda")
targets = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device="cuda")

# 1. Warm-up (確保 CUDA Context 與記憶體已分配完成)
for _ in range(args.warmup_steps):
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 2. 帶有 NVTX 標記的 Profiling 執行
optimizer.zero_grad(set_to_none=True)

# --- Forward Pass ---
nvtx.range_push("forward_pass")
logits = model(inputs)
if args.mode == "fwd":
    torch.cuda.synchronize()
    nvtx.range_pop()
    exit(0)

loss = cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
torch.cuda.synchronize()
nvtx.range_pop()

# --- Backward Pass ---
nvtx.range_push("backward_pass")
loss.backward()
if args.mode == "fwd_bwd":
    torch.cuda.synchronize()
    nvtx.range_pop()
    exit(0)
torch.cuda.synchronize()
nvtx.range_pop()

# --- Optimizer Step ---
nvtx.range_push("optimizer_step")
optimizer.step()
torch.cuda.synchronize()
nvtx.range_pop()