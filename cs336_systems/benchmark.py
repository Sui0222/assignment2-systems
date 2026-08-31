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
parser.add_argument(
    "--memory_profile",
    action="store_true",
    help="Record a CUDA memory history snapshot (for pytorch.org/memory_viz)",
)
parser.add_argument(
    "--dtype",
    choices=["fp32", "fp16", "bf16"],
    default="fp32",
    help="Autocast dtype for the forward pass (backward follows automatically).",
)
args = parser.parse_args()

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
autocast_enabled = args.dtype != "fp32"
autocast_dtype = DTYPE_MAP[args.dtype]

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

    # autocast only wraps the forward pass (+ loss). Backward automatically
    # reuses whatever dtype each op ran forward in -- you should NOT wrap
    # loss.backward() in autocast yourself.
    with torch.autocast(
        device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled
    ):
        if args.mode == "fwd":
            with torch.no_grad():
                logits = model(inputs)
            del logits
            return

        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

    if args.mode == "fwd_bwd":
        loss.backward()
        del logits, loss
        return

    loss.backward()
    optimizer.step()
    del logits, loss


# Warm-up
for _ in range(args.warmup_steps):
    run_step()

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 三選一：memory snapshot > torch.profiler > 純計時
if args.memory_profile:
    # 只記錄 measure 階段，不含 warmup（warmup 有 CUDA context 初始化 /
    # cuDNN autotune 等雜訊，不是我們想分析的東西）
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # 一兩步就夠了，太多步只是浪費 CPU 記憶體記錄歷史，不會有新資訊
    for _ in range(2):
        run_step()

    snapshot_path = f"memory_snapshot_{args.model_size}_{args.dtype}.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\n[SUCCESS] Memory snapshot saved to {snapshot_path}")
    print("Load it at https://pytorch.org/memory_viz to visualize.")

# 如果開啟 --profile 參數，則執行 Profiling
elif args.profile:
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
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(
        f"[{args.model_size} | Mode: {args.mode} | dtype: {args.dtype}] "
        f"Mean: {mean_time:.2f} ms | Std: {std_time:.2f} ms | "
        f"Peak mem: {peak_mem_gb:.2f} GB"
    )

# 清理，避免同一個 process 內連續跑多個 model_size 時顯存殘留 / fragmentation
del model, optimizer, inputs, targets
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()