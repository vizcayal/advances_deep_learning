import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128):
        super().__init__()
        self.patch_conv1 = torch.nn.Conv2d(3, latent_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(latent_dim // 4)
        self.gelu = torch.nn.GELU()
        
        self.patch_conv2 = torch.nn.Conv2d(latent_dim // 4, latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(latent_dim)
        
        self.final_conv = torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = hwc_to_chw(x)  # Convert to (B, C, H, W)
        x = self.gelu(self.bn1(self.patch_conv1(x)))
        x = self.bn2(self.patch_conv2(x))
        #x = self.final_conv(x)  # Patchify step
        return chw_to_hwc(x)  # Convert back to (B, H//patch_size, W//patch_size, latent_dim)



class UnpatchifyLinear(torch.nn.Module):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128):
        super().__init__()
        #self.unpatch_conv1 = torch.nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        #self.bn1 = torch.nn.BatchNorm2d(latent_dim)
        self.gelu = torch.nn.GELU()
        
        self.unpatch_conv2 = torch.nn.Conv2d(latent_dim, latent_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(latent_dim // 4)
        
        self.final_conv = torch.nn.Conv2d(latent_dim // 4, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = hwc_to_chw(x)  # Convert to (B, C, H, W)
        #x = self.gelu(self.bn1(self.unpatch_conv1(x)))
        x = self.gelu(self.bn2(self.unpatch_conv2(x)))
        x = self.final_conv(x)  # Final reconstruction
        return chw_to_hwc(x)  # Convert back to (B, H, W, 3)


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.encoder = PatchifyLinear(patch_size= patch_size, latent_dim= latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)
             

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.decoder = UnpatchifyLinear(patch_size= patch_size, latent_dim= latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
