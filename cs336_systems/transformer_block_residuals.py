import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RotaryEmbedding, TransformerBlock


# XL model configuration. The full model contains 32 Transformer blocks.
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 128
batch_size = 1  # 4
run_backward = True
device = "cuda"

block = TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=RotaryEmbedding(
        dim=d_model // num_heads,
        context_length=context_length,
    ),
).to(device)

# A parameter gradient has the same shape and dtype as its parameter here.
expected_gradient_bytes = sum(parameter.numel() * parameter.element_size() for parameter in block.parameters())
print(f"Expected parameter-gradient memory: {expected_gradient_bytes / 1024**2:.4f} MiB")

# Fuse as much as torch.compile will allow.
block = torch.compile(block, fullgraph=True)
x = torch.randn(
    (batch_size, context_length, d_model),
    device=device,
    requires_grad=True,
)

# Log the tensors saved by autograd for the backward pass.
saved_tensor_bytes = 0


def pack_hook(tensor):
    if isinstance(tensor, torch.nn.Parameter):
        return tensor

    global saved_tensor_bytes
    saved_tensor_bytes += tensor.numel() * tensor.element_size()
    print(f"Saving tensor: shape={tensor.shape}, dtype={tensor.dtype}, grad_fn={tensor.grad_fn}")
    return tensor


def unpack_hook(tensor):
    return tensor


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x)

print(f"Total size of tensors saved by one TransformerBlock: {saved_tensor_bytes / 1024**2:.2f} MiB")

# On a small GPU, first use batch_size=1 and context_length=128 before
# changing run_backward to True.
if run_backward:
    y.sum().backward()
    actual_gradient_bytes = sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in block.parameters() if parameter.grad is not None)
    print(f"Actual parameter-gradient memory: {actual_gradient_bytes / 1024**2:.4f} MiB")

# Release the first computation graph, then reset the hook counter so the next
# result contains only the checkpointed four-block experiment.
del y
saved_tensor_bytes = 0


def two_blocks(x):
    x = block(x)
    x = block(x)
    return x


def four_blocks_checkpoint(x):
    # Checkpoint discards tensors saved inside each two-block group. During
    # backward, it reruns that group's forward pass before computing gradients.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)

print(f"Total size of tensors saved in four TransformerBlocks with checkpointing: {saved_tensor_bytes / 1024**2:.2f} MiB")
