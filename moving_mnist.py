import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import random
from tqdm import tqdm

# ========= Diffusion Process ========= #
class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1).sqrt()
        sqrt_one_minus = (1 - self.alpha_bars[t]).view(-1, 1, 1, 1, 1).sqrt()
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def p_sample(self, x_t, pred_noise, t, t_index):
        beta_t = self.betas[t_index]
        alpha_t = self.alphas[t_index]
        alpha_bar_t = self.alpha_bars[t_index]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)

        mean = coef1 * (x_t - coef2 * pred_noise)
        if t_index == 0:
            return mean
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

# ========= Moving MNIST Dataset ========= #
class MovingMNISTDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path)  # shape: (N, 20, 64, 64)
        self.data = self.data[:, :, None, :, :] / 255.0  # (N, 20, 1, 64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[idx], dtype=torch.float32)
        return sequence

# ========= Dummy Model (Conv-based U-Net Block) ========= #
class SimpleConvPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, noisy_target, cond_frames):
        # 拼接在时间维度（dim=1）
        x_input = torch.cat([cond_frames, noisy_target], dim=1)  # (B, T, 1, H, W)
        x_input = x_input.permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W)

        features = self.encoder(x_input)  # (B, 64, T, H, W)
        out = self.decoder(features)      # (B, 1, T, H, W)

        out = out.permute(0, 2, 1, 3, 4)  # (B, T, 1, H, W)
        return out[:, -noisy_target.size(1):]  # 仅保留目标帧的输出部分

# ========= Training Loop ========= #
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MovingMNISTDataset("moving_mnist.npy")  # 需要预先准备
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleConvPredictor().to(device)
    diffusion = Diffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(500):
        for batch in tqdm(dataloader):
            x = batch.to(device)  # (B, 20, 1, 64, 64)  

            # 随机选一半为条件帧
            B, T, C, H, W = x.shape
            perm = torch.randperm(T)
            cond_idx = perm[:T//2]
            target_idx = perm[T//2:]

            x_cond = x[:, cond_idx]  # (B, T1, 1, H, W)
            x_target = x[:, target_idx]  # (B, T2, 1, H, W)

            t = torch.randint(0, diffusion.timesteps, (B,), device=x.device)
            noise = torch.randn_like(x_target)
            noisy_target = diffusion.q_sample(x_target, t, noise=noise)

            pred_noise = model(noisy_target, x_cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ========= Inference / Sampling ========= #
@torch.no_grad()
def sample():
    model = SimpleConvPredictor().cuda()
    model.load_state_dict(torch.load("ddpm_mnist.pth"))
    model.eval()

    diffusion = Diffusion()
    dataset = MovingMNISTDataset("moving_mnist.npy")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        if i >= 5: break  # 只测试5个样本

        x = batch.cuda()  # (1, 20, 1, 64, 64)
        B, T, C, H, W = x.shape
        perm = torch.randperm(T)
        cond_idx = perm[:T//2]
        target_idx = perm[T//2:]

        x_cond = x[:, cond_idx]
        x_target = x[:, target_idx]

        x_t = torch.randn_like(x_target).cuda()

        for t_index in reversed(range(diffusion.timesteps)):
            t = torch.full((B,), t_index, device=x.device, dtype=torch.long)
            pred_noise = model(x_t, x_cond)
            x_t = diffusion.p_sample(x_t, pred_noise, t, t_index)

        pred = x_t.squeeze(0).squeeze(1).cpu().numpy()  # (T2, H, W)
        gt = x_target.squeeze(0).squeeze(1).cpu().numpy()

        # 保存gif对比
        pred_gif = [((p * 255).clip(0, 255)).astype(np.uint8) for p in pred]
        gt_gif = [((g * 255).clip(0, 255)).astype(np.uint8) for g in gt]

        imageio.mimsave(f"pred_{i}.gif", pred_gif, fps=5)
        imageio.mimsave(f"gt_{i}.gif", gt_gif, fps=5)
        print(f"Sample {i} saved.")

if __name__ == "__main__":
    train()
    sample()
