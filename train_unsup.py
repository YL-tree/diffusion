from torch.utils.data import DataLoader
from mnist_data import MNIST
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import T, forward_add_noise_pair, posterior_probs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======================
# 辅助函数
# ======================
def save_loss_plot(loss_list, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

# ======================
# 训练主程序
# ======================
if __name__ == '__main__':
    EPOCH = 200
    BATCH_SIZE = 200
    K = 10  # MNIST 类别数

    dataset = MNIST()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
    model_path = 'model/unet_unsup.pth'
    loss_plot_path = 'results/unet_unsup_loss.png'

    try:
        model.load_state_dict(torch.load(model_path))
    except:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='none')  # 用 MSE，和 DDPM 一致

    model.train()
    iter_count = 0
    loss_list = []

    for epoch in tqdm(range(EPOCH), desc='Training', unit='epoch'):
        for imgs, _ in dataloader:
            x0 = imgs.to(DEVICE) * 2 - 1
            t = torch.randint(1, T, (x0.size(0),), device=DEVICE).long()

            # 构造 (z_t, z_{t-1}, eps_eff)
            z_t, z_tm1, eps_eff = forward_add_noise_pair(x0, t)

            # 计算类别后验分布
            probs = posterior_probs(model, z_t, z_tm1, t, K, tau=1.0).detach()

            # Soft-EM 损失：对每个类的噪声预测误差，按后验加权
            per_class_losses = []
            for k in range(K):
                yk = torch.full((x0.size(0),), k, dtype=torch.long, device=DEVICE)
                eps_pred = model(z_t, t, yk)
                per_pix = loss_fn(eps_pred, eps_eff)
                w = probs[:, k].view(-1, 1, 1, 1)
                per_class_losses.append((per_pix * w).mean())
            loss = torch.stack(per_class_losses).sum()
            
            # # Hard-EM 损失：直接最大化后验概率
            # k_star = probs.argmax(dim=1)  # [B]
            # eps_pred = model(z_t, t, k_star)
            # loss = loss_fn(eps_pred, eps_eff).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if iter_count % 1000 == 0:
                torch.save(model.state_dict(), model_path)
                entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1).mean().item()
                print(f"epoch {epoch}, iter {iter_count}, loss {loss.item():.4f}, posterior entropy {entropy:.3f}")
            iter_count += 1

    save_loss_plot(loss_list, loss_plot_path)
