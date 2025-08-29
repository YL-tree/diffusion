import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from unet import UNet
from diffusion import T, forward_add_noise_pair
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from inference_unsup import sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# FID计算
# ======================
def save_generated_images(model, num_images=10000, save_dir="results/generated"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch_size = 64
        saved = 0
        while saved < num_images:
            n = min(batch_size, num_images - saved)
            z_T = torch.randn(n, 1, 28, 28).to(DEVICE)
            samples = sample(model, n_samples=n, given_class=None)
            samples = (samples.clamp(-1,1) + 1) / 2.0
            for i in range(n):
                save_image(samples[i], f"{save_dir}/{saved+i:05d}.png")
            saved += n


def save_real_images(num_images=10000, save_dir="results/real"):
    os.makedirs(save_dir, exist_ok=True)
    dataset = MNIST(root="./data", train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x*2 - 1)
                    ]))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    count = 0
    for imgs, _ in loader:
        save_image((imgs+1)/2, f"{save_dir}/{count:05d}.png")
        count += 1
        if count >= num_images:
            break


def compute_fid(real_dir="results/real", gen_dir="results/generated"):
    metrics = calculate_metrics(
        input1=real_dir,
        input2=gen_dir,
        cuda=torch.cuda.is_available(),
        isc=False, fid=True, kid=False,
        verbose=True
    )
    print("FID:", metrics["frechet_inception_distance"])


# ======================
# NLL估计 (ELBO bound)
# ======================
def compute_nll(model, dataloader, T=1000):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x0, _ in dataloader:
            x0 = x0.to(DEVICE) * 2 - 1
            bsz = x0.size(0)
            t = torch.randint(1, T+1, (bsz,), device=DEVICE)
            z_t, _, eps_eff = forward_add_noise_pair(x0, t)
            yk = torch.randint(0, 10, (bsz,), device=DEVICE)  # 随机类
            eps_pred = model(z_t, t, yk)
            mse = torch.mean((eps_pred - eps_eff) ** 2, dim=(1,2,3))
            total_loss += mse.sum().item()
            count += bsz
    nll_est = total_loss / count
    print("Approximate NLL (ELBO bound):", nll_est)
    return nll_est


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    K = 10
    model = UNet(img_channels=1, base_ch=64, channel_mults=(1,2,4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
    model.load_state_dict(torch.load("model/unet_unsup.pth", map_location=DEVICE))
    model.eval()


    # ========== FID ==========
    save_real_images(num_images=5000)
    save_generated_images(model, num_images=5000)
    compute_fid("results/real", "results/generated")

    # ========== NLL ==========
    test_dataset = MNIST(root="./data", train=False, download=True,
                         transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    compute_nll(model, test_loader, T=T)