import argparse
import gc
import timeit
import numpy as np
import torch

from config import CONFIGS
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", choices=CONFIGS.keys(), default="small")
parser.add_argument("--mode", choices=["fwd", "fwd_bwd", "full"], default="full")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument("--measure_steps", type=int, default=10)
parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
args = parser.parse_args()

config = CONFIGS[args.model_size]

model = (
    BasicsTransformerLM(
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
    )
    .cuda()
    .train()
)

optimizer = AdamW(model.parameters(), lr=1e-3)


def get_dummy_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: str = "cuda"
):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


inputs, targets = get_dummy_batch(
    batch_size=args.batch_size,
    seq_len=config.context_length,
    vocab_size=config.vocab_size,
)


def run_step():
    optimizer.zero_grad(set_to_none=True)

    if args.mode == "fwd":
        with torch.no_grad():
            logits = model(inputs)
        del logits
        return

    logits = model(inputs)
    loss = cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()

    if args.mode == "fwd_bwd":
        del logits, loss
        return

    optimizer.step()
    del logits, loss


# Warm-up
for _ in range(args.warmup_steps):
    run_step()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 如果開啟 --profile 參數，則執行 Profiling
if args.profile:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        # 只捕捉 1~2 步即可，避免檔名與記憶體過大
        for _ in range(2):
            run_step()

    # 1. 印出 Console 摘要表（Top 15 最耗時 CUDA Kernel）
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # 2. 匯出 Trace 圖表檔案
    prof.export_chrome_trace("trace.json")
    print("\n[SUCCESS] Trace successfully saved to trace.json")

else:
    # 標準 Benchmark 計時
    timings = []
    for _ in range(args.measure_steps):
        torch.cuda.synchronize()
        start = timeit.default_timer()

        run_step()

        torch.cuda.synchronize()
        end = timeit.default_timer()
        timings.append((end - start) * 1000)

    mean_time = np.mean(timings)
    std_time = np.std(timings)
    print(
        f"[{args.model_size} | Mode: {args.mode}] Mean: {mean_time:.2f} ms | Std: {std_time:.2f} ms"
    )