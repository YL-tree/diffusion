import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DiTBlock(nn.Module):
    def __init__(self, emb_size, nhead):
        super().__init__()

        self.emb_size = emb_size
        self.nhead = nhead

        # conditioning 参数
        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)
        self.alpha1 = nn.Linear(emb_size, emb_size)
        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)

        # layer norm
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        # multi-head self-attention
        self.wq = nn.Linear(emb_size, nhead * emb_size)
        self.wk = nn.Linear(emb_size, nhead * emb_size)
        self.wv = nn.Linear(emb_size, nhead * emb_size)
        self.lv = nn.Linear(nhead * emb_size, emb_size)

        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x, cond):
        # cond: (batch, emb_size)
        gamma1_val = self.gamma1(cond)
        beta1_val = self.beta1(cond)
        alpha1_val = self.alpha1(cond)
        gamma2_val = self.gamma2(cond)
        beta2_val = self.beta2(cond)
        alpha2_val = self.alpha2(cond)

        # layer norm + scale shift
        y = self.ln1(x)
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1)

        # attention
        q = self.wq(y)
        k = self.wk(y)
        v = self.wv(y)

        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 3, 1)
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)

        attn = q @ k / math.sqrt(self.emb_size)
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v

        y = y.permute(0, 2, 1, 3).reshape(y.size(0), y.size(2), self.nhead * self.emb_size)
        y = self.lv(y)

        # scale + residual
        y = x + y * alpha1_val.unsqueeze(1)

        # feed forward
        z = self.ln2(y)
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        z = self.ff(z)
        z = z * alpha2_val.unsqueeze(1)

        return y + z


if __name__ == '__main__':
    dit_block = DiTBlock(emb_size=16, nhead=4)

    x = torch.rand((5, 49, 16))
    cond = torch.rand((5, 16))

    outputs = dit_block(x, cond)
    print(outputs.shape)  # [5, 49, 16]