from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # # Keep the LoRA layers in float32
        # raise NotImplementedError()
        super().__init__(in_features, out_features, bias)
        for param in super().parameters():
          param.requires_grad = False

        self.linear_dtype = torch.float32
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias = False, dtype = self.linear_dtype)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias = False, dtype = self.linear_dtype)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        x_dtype = x.dtype 
        x = x.to(self.linear_dtype)
        output =  super().forward(x) + self.lora_b(self.lora_a(x))
        output = output.to(x_dtype)
        return output


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()

            self.model = torch.nn.Sequential(
                QLoRALinear(in_features = channels, out_features = channels, lora_dim = 20),
                torch.nn.ReLU(),
                QLoRALinear(in_features = channels, out_features = channels, lora_dim = 20),
                torch.nn.ReLU(),
                QLoRALinear(in_features = channels, out_features = channels, lora_dim = 20),

            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()

        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
