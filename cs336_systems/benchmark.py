import argparse
import gc
import timeit
from collections.abc import Callable

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.config import CONFIGS


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=CONFIGS.keys(), default="small")
    parser.add_argument("--mode", choices=["fwd", "fwd_bwd", "full"], default="full")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--memory_profile", action="store_true")
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="fp32")
    return parser.parse_args()


def get_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return inputs, targets


def make_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    vocab_size: int,
    autocast_dtype: torch.dtype,
) -> Callable[[], None]:
    autocast_enabled = autocast_dtype != torch.float32

    def run_step() -> None:
        if mode == "fwd":
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    model(inputs)
            return

        optimizer.zero_grad(set_to_none=True)

        # Autocast selects the forward operators' dtypes. Backward must run
        # outside this context and reuses the dtypes chosen during forward.
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(inputs)
            loss = cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

        loss.backward()
        if mode == "full":
            optimizer.step()

    return run_step


def prepare_measurement() -> None:
    """Remove warm-up garbage and start fresh peak-memory accounting."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def run_memory_profile(
    run_step: Callable[[], None],
    snapshot_path: str,
) -> None:
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        for _ in range(2):
            run_step()
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(snapshot_path)
    finally:
        # Always stop recording, including when a step or snapshot fails.
        torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\n[SUCCESS] Memory snapshot saved to {snapshot_path}")
    print("Load it at https://pytorch.org/memory_viz to visualize.")


def run_torch_profile(
    run_step: Callable[[], None],
    trace_path: str,
) -> None:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(2):
            run_step()

    print(prof.key_averages().table(sort_by="device_time_total", row_limit=15))
    prof.export_chrome_trace(trace_path)
    print(f"\n[SUCCESS] Trace saved to {trace_path}")


def run_benchmark(
    run_step: Callable[[], None],
    measure_steps: int,
) -> tuple[float, float, float]:
    timings_ms = []
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        run_step()
        torch.cuda.synchronize()
        timings_ms.append((timeit.default_timer() - start) * 1_000)

    mean_ms = float(np.mean(timings_ms))
    std_ms = float(np.std(timings_ms))
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    return mean_ms, std_ms, peak_memory_gib


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable GPU.")

    device = torch.device("cuda")
    config = CONFIGS[args.model_size]
    context_length = args.context_length or config.context_length

    model = None
    optimizer = None
    inputs = None
    targets = None
    try:
        model = BasicsTransformerLM(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            context_length=context_length,
        ).to(device)
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        inputs, targets = get_dummy_batch(
            batch_size=args.batch_size,
            seq_len=context_length,
            vocab_size=config.vocab_size,
            device=device,
        )
        run_step = make_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            mode=args.mode,
            vocab_size=config.vocab_size,
            autocast_dtype=DTYPE_MAP[args.dtype],
        )

        for _ in range(args.warmup_steps):
            run_step()
        prepare_measurement()

        run_name = f"{args.model_size}_{args.mode}_{args.dtype}_ctx{context_length}"
        if args.memory_profile:
            run_memory_profile(
                run_step,
                f"memory_snapshot_{run_name}.pickle",
            )
        elif args.profile:
            run_torch_profile(run_step, f"trace_{run_name}.json")
        else:
            mean_ms, std_ms, peak_memory_gib = run_benchmark(run_step, args.measure_steps)
            print(
                f"[{args.model_size} | mode: {args.mode} | dtype: {args.dtype} | "
                f"batch: {args.batch_size} | context: {context_length}] "
                f"Mean: {mean_ms:.2f} ms | Std: {std_ms:.2f} ms | "
                f"Peak memory: {peak_memory_gib:.2f} GiB"
            )
    finally:
        del model, optimizer, inputs, targets
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
