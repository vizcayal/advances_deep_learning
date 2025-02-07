from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        
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
    


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()

            self.model = torch.nn.Sequential(
                LoRALinear(in_features = channels, out_features = channels, lora_dim = 20),
                torch.nn.ReLU(),
                LoRALinear(in_features = channels, out_features = channels, lora_dim = 20),
                torch.nn.ReLU(),
                LoRALinear(in_features = channels, out_features = channels, lora_dim = 20),

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


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net