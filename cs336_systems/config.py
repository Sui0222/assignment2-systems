from dataclasses import dataclass

# 1. 定義配置類別
@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    vocab_size: int = 500
    context_length: int = 512

# 2. 定義 CS336/GPT 系列標準模型規格 mapping
CONFIGS = {
    "small":  ModelConfig(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  ModelConfig(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}