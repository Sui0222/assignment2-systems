import argparse
import gc
from contextlib import AbstractContextManager, nullcontext

import torch
from torch.cuda import nvtx

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.config import CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=CONFIGS.keys(), default="small")
    parser.add_argument("--mode", choices=["fwd", "fwd_bwd", "full"], default="full")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=5)
    return parser.parse_args()


def optional_nvtx_range(name: str, enabled: bool) -> AbstractContextManager[None]:
    if enabled:
        return nvtx.range(name)
    return nullcontext()


def run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    vocab_size: int,
    annotate: bool,
) -> None:
    loss: torch.Tensor | None = None

    with optional_nvtx_range("profile_step", annotate):
        if mode != "fwd":
            optimizer.zero_grad(set_to_none=True)

        with optional_nvtx_range("forward_pass", annotate):
            if mode == "fwd":
                with torch.no_grad():
                    model(inputs)
            else:
                logits = model(inputs)
                loss = cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                )
            torch.cuda.synchronize()

        if mode == "fwd":
            return

        assert loss is not None
        with optional_nvtx_range("backward_pass", annotate):
            loss.backward()
            torch.cuda.synchronize()

        if mode == "fwd_bwd":
            return

        with optional_nvtx_range("optimizer_step", annotate):
            optimizer.step()
            torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires a CUDA-capable GPU.")

    config = CONFIGS[args.model_size]
    context_length = args.context_length or config.context_length
    device = torch.device("cuda")

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
    inputs = torch.randint(0, config.vocab_size, (args.batch_size, context_length), device=device)
    targets = torch.randint(0, config.vocab_size, (args.batch_size, context_length), device=device)

    try:
        for _ in range(args.warmup_steps):
            run_step(model, optimizer, inputs, targets, args.mode, config.vocab_size, annotate=False)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        run_step(model, optimizer, inputs, targets, args.mode, config.vocab_size, annotate=True)
        print(
            f"Profiled {args.model_size} {args.mode}: "
            f"batch={args.batch_size}, context={context_length}"
        )
    finally:
        del model, optimizer, inputs, targets
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
