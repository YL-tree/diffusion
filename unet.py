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


# ---- Building Blocks ----
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = _group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # add time conditioning (broadcast to HxW)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)
        # stride-2 downsample keeps sizes integral with padding=1
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x  # store pre-downsample features for skip-connection
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        # first upsample latent to match (roughly) spatial dims of the skip path
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        # then fuse with skip and refine
        self.res1 = ResBlock(out_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # in case of odd sizes (e.g., 28->14->7->4), align spatial dims exactly
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


# ---- U-Net ----
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

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # optional class embedding (additive to time emb)
        self.class_emb = nn.Embedding(num_classes, time_emb_dim) if num_classes is not None else None

        # input stem
        self.init_conv = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)

        # encoder
        downs: List[DownBlock] = []
        ch = base_ch
        self.skip_chs: List[int] = []
        for mult in channel_mults:
            out_ch = base_ch * mult
            downs.append(DownBlock(ch, out_ch, time_emb_dim))
            self.skip_chs.append(out_ch)
            ch = out_ch
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.mid1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # decoder (mirror of encoder)
        ups: List[UpBlock] = []
        for skip_ch in reversed(self.skip_chs):
            out_ch = skip_ch  # usually mirror encoder width
            ups.append(UpBlock(ch, skip_ch, out_ch, time_emb_dim))
            ch = out_ch
        self.ups = nn.ModuleList(ups)

        # output head
        self.final_norm = _group_norm(ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(ch, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # build conditioning
        t_emb = self.time_mlp(t)
        if self.class_emb is not None and y is not None:
            t_emb = t_emb + self.class_emb(y)

        # encoder
        x = self.init_conv(x)
        skips: List[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        # bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # decoder
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)

        # output
        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x


if __name__ == "__main__":
    # quick sanity check
    B, C, H, W = 8, 1, 28, 28
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, 10, (B,))

    model = UNet(img_channels=C, base_ch=64, channel_mults=(1, 2, 4), time_emb_dim=128, num_classes=10)
    with torch.no_grad():
        out = model(x, t, y)
    print("out:", out.shape)  # should be (B, C, H, W)
