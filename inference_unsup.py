import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from unet import UNet
from diffusion import T, betas, alphas, alphas_cumprod, variance, extract

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======================
# 反向采样过程（给定类别）
# ======================
@torch.no_grad()
def p_sample(model, z_t, t, y_class):
    """
    一个反向采样步：
    z_{t-1} = 1/sqrt(alpha_t) * (z_t - beta_t/sqrt(1 - \bar{alpha}_t) * eps_pred) + sigma_t * noise
    """
    b_t = extract(betas, t, z_t.shape)
    a_t = extract(alphas, t, z_t.shape)
    a_bar_t = extract(alphas_cumprod, t, z_t.shape)

    eps_pred = model(z_t, t, y_class)

    # 均值项
    mu = (1.0 / torch.sqrt(a_t)) * (z_t - (b_t / torch.sqrt(1.0 - a_bar_t)) * eps_pred)

    if t[0] > 0:
        var_t = extract(variance, t, z_t.shape)
        noise = torch.randn_like(z_t)
        z_prev = mu + torch.sqrt(var_t) * noise
    else:
        z_prev = mu
    return z_prev

@torch.no_grad()
def sample(model, n_samples=16, given_class=None, K=10):
    """
    从模型采样
    - given_class: 如果指定，则所有样本用该类；否则从均匀先验采样类
    """
    z_t = torch.randn(n_samples, 1, 28, 28).to(DEVICE)  # 初始噪声
    for t_step in reversed(range(T)):
        t_tensor = torch.full((n_samples,), t_step, device=DEVICE, dtype=torch.long)
        if given_class is None:
            y_class = torch.randint(0, K, (n_samples,), device=DEVICE)
        else:
            y_class = torch.full((n_samples,), given_class, device=DEVICE)
        z_t = p_sample(model, z_t, t_tensor, y_class)
    return z_t

# ======================
# 可视化函数
# ======================
def save_images(samples, path, nrow=4):
    samples = (samples.clamp(-1,1) + 1) / 2.0  # [-1,1] -> [0,1]
    samples = samples.cpu()
    grid = torch.zeros(1, samples.size(2)*nrow, samples.size(3)*nrow)
    idx = 0
    for i in range(nrow):
        for j in range(nrow):
            grid[:, i*samples.size(2):(i+1)*samples.size(2), j*samples.size(3):(j+1)*samples.size(3)] = samples[idx]
            idx += 1
    plt.imshow(grid.squeeze(0), cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    K = 10
    model = UNet(img_channels=1, base_ch=64, channel_mults=(1,2,4), time_emb_dim=128, num_classes=K).to(DEVICE)
    model.load_state_dict(torch.load("model/unet_unsup.pth", map_location=DEVICE))
    model.eval()

    os.makedirs("samples", exist_ok=True)

    # 1. 不指定类别（从先验采样）
    samples = sample(model, n_samples=16, given_class=None, K=K)
    save_images(samples, "samples/sample_random.png")

    # 2. 指定每个类别采样（可检查每类学到了什么）
    for cls in range(K):
        samples = sample(model, n_samples=16, given_class=cls, K=K)
        save_images(samples, f"samples/sample_class_{cls}.png")
