import argparse
import timeit
import torch
import numpy as np

from config import CONFIGS  # 導入配置字典
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

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

model = BasicsTransformerLM(
    d_model=config.d_model,
    d_ff=config.d_ff,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    vocab_size=config.vocab_size,
    context_length=config.context_length,
).cuda()

optimizer = AdamW(model.parameters(), lr=1e-3)

# 3. 隨機假數據生成函數
def get_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: str = "cuda"):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y

inputs, targets = get_dummy_batch(
    batch_size=args.batch_size, 
    seq_len=config.context_length, 
    vocab_size=config.vocab_size
)

# 4. 定義單步執行邏輯
def run_step():
    optimizer.zero_grad()
    logits = model(inputs)
    if args.mode == "fwd":
        with torch.no_grad():
            logits = model(inputs)
        return
    # 在 PyTorch 的 .view() 語法中，-1 代表「自動推導維度（Inferred Dimension）」。
    loss = cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    loss.backward()
    if args.mode == "fwd_bwd":
        return
    optimizer.step()

    # 關鍵：清理局部張量引用
    del logits, loss

# 5. GPU 預熱階段 (Warm-up)
for _ in range(args.warmup_steps):
    run_step()

torch.cuda.synchronize()

# 6. 正式計時階段 (Measurement)
timings = []
for _ in range(args.measure_steps):
    torch.cuda.synchronize()
    start = timeit.default_timer()

    run_step()

    torch.cuda.synchronize()
    end = timeit.default_timer()
    timings.append((end - start) * 1000) # 轉成 ms

# 測試完成後手動清空快取
torch.cuda.empty_cache()

# 7. 數據統計與輸出 (方便回答 Deliverable b)
mean_time = np.mean(timings)
std_time = np.std(timings)
print(f"[{args.model_size} | Mode: {args.mode}] Mean: {mean_time:.2f} ms | Std: {std_time:.2f} ms")