import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer

# --- MODEL DEFINITION ---
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()

        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base


# --- LOAD MODEL ONCE (IMPORTANT) ---
device = torch.device("cpu")

model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=32,
    upscale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path="realesr-general-x4v3.pth",
    model=model,
    tile=800,
    tile_pad=32,
    pre_pad=0,
    half=False,
    device=device
)


def enhance_image(pil_image: Image.Image, outscale: int = 2) -> Image.Image:
    img_np = np.array(pil_image.convert("RGB"))
    output, _ = upsampler.enhance(img_np, outscale=outscale)
    return Image.fromarray(output)
