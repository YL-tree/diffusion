import torch
from torch import nn
import torch.nn.functional as F
from Time_emb import TimeEmbedding

from typing import List, Optional, Tuple

# ---- Helpers ----
def _group_norm(c: int, max_groups: int = 32) -> nn.GroupNorm:
    """Create a GroupNorm whose group count divides channels.
    Falls back to fewer groups if needed.
    """
    groups = min(max_groups, c)
    while c % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, c)


# ---- Building Blocks with AdaGN-style conditioning ----
# --- ResBlock (Modified) ---
class ResBlock(nn.Module):
    # The only change is the cond_emb_dim, the rest of the logic is the same
    def __init__(self, in_ch: int, out_ch: int, cond_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        # ... (norm1, act1, conv1 are the same) ...
        self.norm1 = _group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # IMPORTANT: The input dimension here will be larger after concatenation
        self.cond_proj1 = nn.Linear(cond_emb_dim, out_ch * 2)
        self.cond_proj2 = nn.Linear(cond_emb_dim, out_ch * 2)

        # ... (the rest of the __init__ and forward method are identical to your original code) ...
        self.norm2 = _group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # This forward pass does not need to change at all.
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        cs1 = self.cond_proj1(cond_emb)
        scale1, shift1 = cs1.chunk(2, dim=1)
        h = h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]

        h = self.norm2(h)
        cs2 = self.cond_proj2(cond_emb)
        scale2, shift2 = cs2.chunk(2, dim=1)
        h = h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_emb_dim: int):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, cond_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, cond_emb_dim)
        # stride-2 downsample (padding keeps sizes integral)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, cond_emb)
        x = self.res2(x, cond_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_emb_dim: int):
        super().__init__()
        # transpose conv upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        # then fuse with skip and refine
        self.res1 = ResBlock(out_ch + skip_ch, out_ch, cond_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, cond_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # align spatial dims if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, cond_emb)
        x = self.res2(x, cond_emb)
        return x


# ---- U-Net with stronger class conditioning ----
class UNet(nn.Module):
    def __init__(
        self,
        img_channels: int = 1,
        base_ch: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 128,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.time_emb_dim = time_emb_dim

        # time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # class embedding
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)
            # concat 后过一层 MLP，再压回 time_emb_dim
            self.cond_mlp = nn.Sequential(
                nn.Linear(2 * time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.class_emb = None
            self.cond_mlp = None

        # input stem
        self.init_conv = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)

        # encoder
        downs: List[DownBlock] = []
        ch = base_ch
        self.skip_chs: List[int] = []
        for mult in channel_mults:
            out_ch = base_ch * mult
            downs.append(DownBlock(ch, out_ch, time_emb_dim))  # cond_emb_dim = time_emb_dim
            self.skip_chs.append(out_ch)
            ch = out_ch
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.mid1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # decoder
        ups: List[UpBlock] = []
        for skip_ch in reversed(self.skip_chs):
            out_ch = skip_ch
            ups.append(UpBlock(ch, skip_ch, out_ch, time_emb_dim))
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        # output head
        self.final_norm = _group_norm(ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # time embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # class embedding
        if self.class_emb is not None and y is not None:
            c_emb = self.class_emb(y)  # [B, time_emb_dim]
            cond = torch.cat([t_emb, c_emb], dim=1)  # [B, 2*dim]
            cond = self.cond_mlp(cond)  # [B, time_emb_dim]
        else:
            cond = t_emb

        # encoder
        x = self.init_conv(x)
        skips: List[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x, cond)
            skips.append(skip)

        # bottleneck
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)

        # decoder
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, cond)

        # output
        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x


if __name__ == "__main__":
    # quick sanity check
    B, C, H, W = 8, 1, 28, 28
    x = torch.randn(B, C, H, W)
    # t should be integer time indices; create a toy tensor in range [0, 1000)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 10, (B,))

    model = UNet(img_channels=C, base_ch=64, channel_mults=(1, 2, 4), time_emb_dim=128, num_classes=10)
    with torch.no_grad():
        out = model(x, t, y)
    print("out:", out.shape)  # should be (B, C, H, W)
