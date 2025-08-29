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

def save_entropy_plot(entropy_list, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(entropy_list, label='Training Entropy')
    plt.xlabel('Iterations')
    plt.ylabel('Entropy')
    plt.title('Training Entropy Over Time')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
# ======================
# 训练主程序
# ======================
if __name__ == '__main__':
    EPOCH = 3
    BATCH_SIZE = 200
    K = 10  # MNIST 类别数
    # 新增参数：从先验中生成标签的概率
    unlabeled_prob = 0.5

    dataset = MNIST()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
    model_path = 'model/unet_semisup_0.5.pth'
    loss_plot_path = 'results/unet_semisup_0.5_loss.png'
    entropy_plot_path = 'results/unet_semisup_0.5_entropy.png'

    try:
        model.load_state_dict(torch.load(model_path, weights_only=False))
    except FileNotFoundError:
        print("Model file not found, starting training from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}, starting from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='none')  # 用 MSE，和 DDPM 一致

    model.train()
    iter_count = 0
    loss_list = []
    entropy_list = []
        
    for epoch in tqdm(range(EPOCH), desc='Training', unit='epoch'):
        for imgs, labels in dataloader:
            x0 = imgs.to(DEVICE) * 2 - 1
            labels = labels.to(DEVICE)
            t = torch.randint(1, T, (x0.size(0),), device=DEVICE).long()
            
            z_t, z_tm1, eps_eff = forward_add_noise_pair(x0, t)

            # 随机选择哪些样本被视为无标签
            unlabeled_mask = torch.rand(x0.size(0), device=DEVICE) < unlabeled_prob
            labeled_mask = ~unlabeled_mask

            total_loss = 0
            
            # 1. Soft-EM 损失：处理无标签的样本
            if unlabeled_mask.any():
                # E步：计算无标签样本的后验概率
                with torch.no_grad():
                    unlabeled_probs = posterior_probs(
                        model,
                        z_t[unlabeled_mask],
                        z_tm1[unlabeled_mask],
                        t[unlabeled_mask],
                        K,
                        eps_eff,
                        tau=0.5
                    )
                # M步：计算无标签样本的损失，按后验加权
                unlabeled_batch_size = unlabeled_mask.sum()
                
                z_t_tiled = z_t[unlabeled_mask].repeat_interleave(K, dim=0)
                t_tiled = t[unlabeled_mask].repeat_interleave(K, dim=0)
                y_unlabeled_tiled = torch.arange(K, device=DEVICE).repeat(unlabeled_batch_size)
                
                eps_pred_tiled = model(z_t_tiled, t_tiled, y_unlabeled_tiled)
                
                eps_pred_reshaped = eps_pred_tiled.view(unlabeled_batch_size, K, *eps_pred_tiled.shape[1:])
                eps_eff_expanded = eps_eff[unlabeled_mask].unsqueeze(1).expand_as(eps_pred_reshaped)
                
                per_class_loss = loss_fn(eps_pred_reshaped, eps_eff_expanded)
                per_class_loss = per_class_loss.sum(dim=(2, 3, 4))
                
                soft_em_loss = (per_class_loss * unlabeled_probs).sum(dim=1).mean()
                
                # 将 Soft-EM 损失累加到总损失中
                total_loss += soft_em_loss * unlabeled_batch_size
            
            # 2. 标准 MSE 损失：处理有标签的样本
            if labeled_mask.any():
                eps_pred_labeled = model(z_t[labeled_mask], t[labeled_mask], labels[labeled_mask].to(DEVICE))
                labeled_loss = loss_fn(eps_pred_labeled, eps_eff[labeled_mask]).mean()
                
                # 将有标签损失累加到总损失中
                total_loss += labeled_loss * labeled_mask.sum()

            # 3. 对所有样本的损失求平均
            total_loss = total_loss / x0.size(0)


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.item())

            if iter_count % 1000 == 0:
                torch.save(model.state_dict(), model_path)
                with torch.no_grad():
                    # 在这里使用完整的批次来计算熵
                    all_probs = posterior_probs(model, z_t, z_tm1, t, K, eps_eff, tau=0.5)
                    entropy = -(all_probs * (all_probs.clamp_min(1e-12).log())).sum(dim=1).mean().item()
                    entropy_list.append(entropy)
                print(f"epoch {epoch}, iter {iter_count}, loss {total_loss.item():.4f}, posterior entropy {entropy:.3f}")
            iter_count += 1
            
    save_loss_plot(loss_list, loss_plot_path)
    save_entropy_plot(entropy_list, entropy_plot_path)