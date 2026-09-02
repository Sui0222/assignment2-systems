# CS336 Assignment 2 Checklist

依照 `cs336_assignment2_systems.pdf` 的順序完成。每一題應同時檢查程式、實驗結果、圖表、文字回答與測試（若題目要求）。

## 2. Profiling and Benchmarking

- [ ] 1. `benchmarking_script` — Benchmarking Script（4 分）
  - [ ] 完成 end-to-end benchmark 程式
  - [ ] 記錄 forward、backward 與 optimizer step 的 timing
  - [ ] 報告 10 次 measurement 的 mean/std
  - [ ] 比較 0、1、2 與正式 warm-up steps 的結果

- [ ] 2. `nsys_profile` — Nsight Systems Profiling（5 分）
  - [ ] Profile handout 指定的兩個 model sizes
  - [ ] 比較 Nsight 與 Python timing
  - [ ] 找出 forward 與 forward-backward 最耗時的 CUDA kernels
  - [ ] 分析非矩陣乘法 kernels
  - [ ] 分析完整 training step 中各類 kernel 的時間比例
  - [ ] 比較 attention softmax 與 matrix multiplication

- [x] 3. `mixed_precision_accumulation` — Mixed-Precision Accumulation（1 分）
  - [x] 執行 handout 程式
  - [x] 撰寫 2–3 句精度分析

- [ ] 4. `benchmarking_mixed_precision` — Benchmarking Mixed Precision（2 分）
  - [ ] 記錄 inputs、layers、loss 與 gradients 的 dtype
  - [ ] 解釋 RMSNorm/LayerNorm 的精度敏感性
  - [ ] 比較 FP32、FP16 與 BF16 timing
  - [ ] 分析 model size 增加時的趨勢

- [ ] 5. `memory_profiling` — Memory Profiling（4 分）
  - [ ] 產生 XL inference active-memory timeline
  - [ ] 產生 XL full-step active-memory timeline
  - [ ] 建立 context-length sweep 的 forward/full peak-memory 表
  - [ ] 比較 mixed precision 的 forward/full peak memory
  - [ ] 推導 XL residual-stream activation tensor 大小
  - [ ] 找出 memory timeline 中最大的 allocations 與來源
  - [ ] 使用 Nsight 分析單一 TransformerBlock 保存的 residuals
  - [ ] 列出最大的 5 個 contributing operations
  - [ ] 推導單一 TransformerBlock gradients 的記憶體大小

## 3. Single-GPU Memory

以下段落沒有獨立 Problem 編號，但需要理解：

- [ ] Autograd residuals / saved tensors
- [ ] Operator fusion
- [ ] Activation checkpointing
- [ ] Recomputation

- [ ] 6. `gradient_checkpointing` — Memory-Optimal Gradient Checkpointing（4 分）
  - [ ] 說明 memory-optimal recursive checkpointing 策略
  - [ ] 推導 peak activation memory 與 compute complexity
  - [ ] 提供簡短 code sketch
  - [ ] 對 XL、batch 4、context 2048 實測 checkpoint block size
  - [ ] 比較最佳 block size 與相鄰的較小/較大設定

## 4. GPU Kernels

- [ ] 7. `pytorch_attention` — PyTorch Attention Benchmarking（2 分）
  - [ ] 建立 attention-only benchmark
  - [ ] 按 handout 指定的 embedding dimensions 與 sequence lengths sweep
  - [ ] 記錄 forward/backward timing 或 OOM
  - [ ] 對一個 OOM configuration 做 memory accounting
  - [ ] 分析 saved-for-backward memory 如何隨 sequence length 改變

- [ ] 8. `torch_compile` — Torch Compile（2 分）
  - [ ] 比較 compiled/uncompiled attention forward/backward
  - [ ] 比較 compiled/uncompiled Transformer end-to-end benchmark

### Triton Weighted Sum 教學

- [ ] 理解 Triton program instance、tile 與 block pointer
- [ ] 完成 Weighted Sum forward kernel
- [ ] 完成 Weighted Sum backward kernel
- [ ] 理解 `torch.autograd.Function` 如何呼叫 Triton kernel

- [ ] 9. `flash_forward` — FlashAttention-2 Forward Pass（15 分）
  - [ ] 理解 tiling、online softmax 與 log-sum-exp recurrence
  - [ ] 完成純 PyTorch tiled forward
  - [ ] 通過 `test_flash_forward_pass_pytorch`
  - [ ] 完成 Triton forward kernel
  - [ ] 通過 `test_flash_forward_pass_triton`
  - [ ] 加入 causal masking
  - [ ] 保存 backward 所需 tensors 與 `is_causal`

- [ ] 10. `flash_backward` — FlashAttention-2 Backward Pass（5 分）
  - [ ] 理解 handout 的 backward equations
  - [ ] 計算並使用 D vector
  - [ ] 使用 PyTorch 與 `torch.compile` 完成 backward
  - [ ] 通過 `test_flash_backward`

- [ ] 11. `flash_benchmarking` — FlashAttention-2 Benchmarking（5 分）
  - [ ] 使用 `triton.testing.do_bench`
  - [ ] 比較 PyTorch 與 Triton forward latency
  - [ ] 比較 PyTorch 與 Triton backward latency
  - [ ] 比較 end-to-end forward-backward latency
  - [ ] 按 handout 指定的 sequence length、embedding dimension 與 dtype sweep

### Optional

- [ ] Triton FlashAttention backward kernel

## 5. Distributed Data Parallel Training

- [ ] 12. `distributed_communication_single_node` — Distributed Communication (Single Node)（5 分）
  - [ ] 建立單機多 GPU all-reduce benchmark
  - [ ] Sweep handout 指定的 backend、world size 與 tensor size
  - [ ] 繪製圖表或整理表格
  - [ ] 撰寫 2–3 句分析

- [ ] 13. `naive_ddp` — Naïve DDP（5 分）
  - [ ] 在初始化時同步 rank 0 參數
  - [ ] backward 後逐參數 all-reduce gradients
  - [ ] 正確平均 gradients
  - [ ] 完成 adapters
  - [ ] 通過 `tests/test_ddp.py`

- [ ] 14. `naive_ddp_benchmarking` — Naïve DDP Benchmarking（3 分）
  - [ ] 記錄 handout 指定 GPU 設定的 iteration time
  - [ ] 記錄 gradient communication time
  - [ ] 描述 benchmark setup

- [ ] 15. `minimal_ddp_flat_benchmarking` — Minimal DDP with Flat Gradients Benchmarking（2 分）
  - [ ] Flatten 全部 gradients 後只做一次 all-reduce
  - [ ] 記錄 iteration 與 communication time
  - [ ] 與逐參數 all-reduce 比較

- [ ] 16. `ddp_overlap_individual_parameters` — DDP with Overlapping Individual Parameters（5 分）
  - [ ] 實作 DDP wrapper
  - [ ] 註冊 gradient hooks
  - [ ] gradient ready 時啟動 asynchronous all-reduce
  - [ ] backward 後等待 communication 完成並平均 gradients
  - [ ] 正確處理 tied weights 與 `requires_grad=False`
  - [ ] 完成 adapters
  - [ ] 多次通過 `tests/test_ddp.py`

- [ ] 17. `ddp_overlap_individual_parameters_benchmarking` — Overlap Benchmarking（1 分）
  - [ ] 記錄 overlap DDP iteration time
  - [ ] 與 naïve DDP 比較
  - [ ] 產生 naïve DDP Nsight screenshot
  - [ ] 產生 overlap DDP Nsight screenshot

## 6. Optimizer State Sharding

- [ ] 18. `optimizer_state_sharding` — Optimizer State Sharding（15 分）
  - [ ] 實作 sharded optimizer wrapper
  - [ ] 在 ranks 間分配 parameters/optimizer states
  - [ ] local optimizer 只更新所屬 shard
  - [ ] step 後同步更新的 parameters
  - [ ] 正確實作 `zero_grad` 與 `add_param_group`
  - [ ] 完成 adapter
  - [ ] 通過 `tests/test_sharded_optimizer.py`

- [ ] 19. `optimizer_state_sharding_accounting` — Optimizer State Sharding Accounting（5 分）
  - [ ] 比較有/無 sharding 的 peak memory
  - [ ] 拆解 parameters、gradients、optimizer states 等記憶體
  - [ ] 比較 iteration time
  - [ ] 比較本實作與 ZeRO Stage 1

## 7. Fully-Sharded Data Parallel

- [ ] 20. `fsdp` — Fully-Sharded Data Parallel（15 分）
  - [ ] 對指定 module weights 做 sharding
  - [ ] forward/backward 前 all-gather weights
  - [ ] backward 後 reduce-scatter gradients
  - [ ] 維持 FP32 master weights
  - [ ] 支援 mixed-precision communication/compute
  - [ ] gather 完整 parameters/state dict
  - [ ] 完成 adapters
  - [ ] 多次通過 `tests/test_fsdp.py`

- [ ] 21. `fsdp_accounting` — FSDP Accounting（5 分）
  - [ ] 推導預期 peak-memory 節省
  - [ ] Profile 雙 GPU XL model
  - [ ] 分析 weight all-gather 是否及時完成
  - [ ] 提供 Nsight screenshots 與 timing

## 8. Analyzing Parallelism Strategies

- [ ] 22. `alternate_ring_all_reduce` — Alternate Ring All-Reduce（1 分）
  - [ ] 推導 alternate ring all-reduce communication time

- [ ] 23. `data_parallel_calcs` — Data Parallel Calculations（3 分）
  - [ ] 推導 compute time
  - [ ] 推導 backward communication time
  - [ ] 推導 communication bottleneck 條件

- [ ] 24. `fsdp_calcs` — Fully-Sharded Data Parallel Calculations（3 分）
  - [ ] 推導 forward/backward communication volume
  - [ ] 推導 forward/backward communication time
  - [ ] 推導 bottleneck 條件

- [ ] 25. `tp_calcs` — Tensor Parallel Calculations（4 分）
  - [ ] 寫出 forward/backward tensor-parallel equations
  - [ ] 推導 forward/backward communication volume與時間
  - [ ] 推導 bottleneck 條件

- [ ] 26. `fsdp_tp_calcs` — 2D Parallelism Calculations（6 分）
  - [ ] 推導 FSDP + TP forward compute time
  - [ ] 推導可重疊 collective 下的 communication time
  - [ ] 推導 bottleneck 條件
  - [ ] 推導最佳 FSDP/TP 配置條件

## 9. Leaderboard

- [ ] 27. `leaderboard` — Fastest Training Step（10 分）
  - [ ] 在兩張 B200、batch size 2 上執行完整 training step
  - [ ] 包含 forward、loss、backward 與 AdamW update
  - [ ] 記錄最佳 wall-clock time
  - [ ] 確認編譯與 autotuning 沒有污染正式 timing

## 最終提交檢查

- [ ] 所有要求的表格已放入報告
- [ ] 所有要求的 screenshots 已放入報告
- [ ] 所有文字小問均已回答
- [ ] Attention tests 通過
- [ ] DDP tests 多次通過
- [ ] Sharded optimizer tests 通過
- [ ] FSDP tests 多次通過
- [ ] 執行 `./test_and_make_submission.sh`
- [ ] 檢查產生的 submission archive
