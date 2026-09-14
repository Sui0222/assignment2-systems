import gc

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.config import CONFIGS


# Local smoke test. For the final server experiment, use:
# model_size = "xl", batch_size = 4, context_length = 2048
# group_sizes = (4, 6, 8)
model_size = "xl"
batch_size = 4
context_length = 2048
group_sizes = (1,2, 3)
device = "cuda"


def checkpointed_forward(model, inputs, group_size):
    x = model.token_embeddings(inputs)

    for start in range(0, len(model.layers), group_size):
        end = min(start + group_size, len(model.layers))

        def run_group(x, start=start, end=end):
            for layer in model.layers[start:end]:
                x = layer(x)
            return x

        x = checkpoint(run_group, x, use_reentrant=False)

    x = model.ln_final(x)
    return model.lm_head(x)


def measure_peak_memory(model, inputs, targets, group_size=None):
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if group_size is None:
        logits = model(inputs)
        label = "no checkpoint"
    else:
        logits = checkpointed_forward(model, inputs, group_size)
        label = f"group size {group_size}"

    loss = cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    loss.backward()
    torch.cuda.synchronize()

    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{label}] loss: {loss.item():.4f} | peak memory: {peak_memory_gib:.2f} MiB")

torch.manual_seed(0)
config = CONFIGS[model_size]

model = BasicsTransformerLM(
    d_model=config.d_model,
    d_ff=config.d_ff,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    vocab_size=config.vocab_size,
    context_length=context_length,
).to(device)
model.train()

inputs = torch.randint(
    0,
    config.vocab_size,
    (batch_size, context_length),
    device=device,
)
targets = torch.randint(
    0,
    config.vocab_size,
    (batch_size, context_length),
    device=device,
)

measure_peak_memory(model, inputs, targets)
for group_size in group_sizes:
    measure_peak_memory(model, inputs, targets, group_size)
