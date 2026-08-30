import argparse
import gc
import timeit
import numpy as np
import torch

from config import CONFIGS  # 導入配置字典
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# 1. 解析命令行參數
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", choices=CONFIGS.keys(), default="small")
parser.add_argument("--mode", choices=["fwd", "fwd_bwd", "full"], default="full")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--warmup_steps", type=int, default=5)
parser.add_argument("--measure_steps", type=int, default=10)
args = parser.parse_args()

# 2. 取得超參數物件與初始化模型
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


# 3. 隨機假數據生成函數
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


# 4. 定義單步執行邏輯（已修復早期 Return 與計算圖洩漏問題）
def run_step():
    optimizer.zero_grad(set_to_none=True)

    # Mode 1: 僅前向傳播 (Pure Forward)
    if args.mode == "fwd":
        with torch.no_grad():
            logits = model(inputs)
        del logits
        return

    # Mode 2 & 3: 需要計算梯度的前向傳播
    logits = model(inputs)
    loss = cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()

    # Mode 2: 前向 + 反向傳播 (Forward + Backward)
    if args.mode == "fwd_bwd":
        del logits, loss
        return

    # Mode 3: 完整訓練步 (Full Training Step)
    optimizer.step()
    del logits, loss


# 5. GPU 預熱階段 (Warm-up)
for _ in range(args.warmup_steps):
    run_step()

# 強制觸發 GC 與顯存清理
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 6. 正式計時階段 (Measurement)
timings = []
for _ in range(args.measure_steps):
    torch.cuda.synchronize()
    start = timeit.default_timer()

    run_step()

    torch.cuda.synchronize()
    end = timeit.default_timer()
    timings.append((end - start) * 1000)  # 轉成 ms

# 7. 數據統計與輸出
mean_time = np.mean(timings)
std_time = np.std(timings)
print(
    f"[{args.model_size} | Mode: {args.mode}] Mean: {mean_time:.2f} ms | Std: {std_time:.2f} ms"
)