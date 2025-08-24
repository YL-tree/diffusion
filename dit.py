import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dit_block import DiTBlock  # 上面改好的 DiTBlock
from Time_emb import TimeEmbedding

class DiT(nn.Module):
    def __init__(self, img_size, patch_size, channel, emb_size, label_num, dit_num, head):
        super().__init__()

        self.patch_size = patch_size
        self.patch_count = img_size // patch_size
        self.channel = channel

        # patch embedding
        self.conv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel * patch_size ** 2,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.patch_emb = nn.Linear(channel * patch_size ** 2, emb_size)
        self.patch_pos_emb = nn.Parameter(torch.randn(1, self.patch_count ** 2, emb_size))

        # time emb
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # label emb
        self.label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)

        # DiT Blocks
        self.dits = nn.ModuleList([DiTBlock(emb_size, head) for _ in range(dit_num)])

        # layer norm
        self.ln = nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear = nn.Linear(emb_size, channel * patch_size ** 2)

    def forward(self, x, t, y):
        # label + time emb
        y_emb = self.label_emb(y)
        t_emb = self.time_emb(t)
        cond = y_emb + t_emb

        # patch emb
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))
        x = self.patch_emb(x) + self.patch_pos_emb

        # dit blocks
        for dit in self.dits:
            x = dit(x, cond)

        # layer norm
        x = self.ln(x)

        # back to image
        x = self.linear(x)
        x = x.view(x.size(0), self.patch_count, self.patch_count, self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            x.size(0), self.channel, self.patch_count * self.patch_size, self.patch_count * self.patch_size
        )
        return x


# -----------------------
# Example usage / Dummy run
# -----------------------
if __name__ == "__main__":
    B = 2
    C = 4
    H = W = 32
    model = DiT(img_size=H, patch_size=1, channel=C, emb_size=256, label_num=10, dit_num=6, head=8)
    latents = torch.randn(B, C, H, W)
    timesteps = torch.randint(0, 1000, (B,))
    labels = torch.randint(0, 10, (B,))
    pred_noise = model(latents, timesteps, labels)
    print("pred_noise", pred_noise.shape)  # [B, C, H, W]
