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
def save_loss_plot(loss_list, plot_name):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label=f'Training {plot_name}')
    plt.xlabel('Iterations')
    plt.ylabel(f'{plot_name}')
    plt.title(f'Training {plot_name} Over Time')
    plt.legend()
    plt.savefig(f'results/unet_{plot_name}.png')
    plt.close()

# ======================
# 训练主程序
# ======================
if __name__ == '__main__':
    EPOCH = 100
    BATCH_SIZE = 200
    K = 10  # MNIST 类别数
    # 新增参数：从先验中生成标签的概率
    unlabeled_prob = 0

    dataset = MNIST()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 步骤二：定义网格搜索空间 ---
    # (根据您在步骤一中观察到的损失量级来设定)
    lambda_grid = [0.02, 0.2, 2.0] 
    best_lambda = None
    best_val_accuracy = 0.0
    best_lambda = None
    
    for lambda_val in lambda_grid:
        print(f"Training with lambda = {lambda_val}")
        model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
        model_path = f'model/unet_{lambda_val}.pth'
        loss_plot_name = f'loss_{lambda_val}'

        # --- 步骤三：训练模型 ---
        try:
            model.load_state_dict(torch.load(model_path, weights_only=False))
        except FileNotFoundError:
            print("Model file not found, starting training from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}, starting from scratch.")

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss(reduction='none')  # 用 MSE，和 DDPM 一致
        
        model.train()
        iter_count = 0
        loss_list = []
        mse_loss_list = []
        entropy_list = []
            
        for epoch in tqdm(range(EPOCH), desc='Training', unit='epoch'):
            for imgs, labels in dataloader:
                x0 = imgs.to(DEVICE) * 2 - 1
                labels = labels.to(DEVICE)
                t = torch.randint(1, T + 1, (x0.size(0),), device=DEVICE).long()
                
                z_t, z_tm1, eps_eff = forward_add_noise_pair(x0, t)

                unlabeled_mask = torch.rand(x0.size(0), device=DEVICE) < unlabeled_prob
                labeled_mask = ~unlabeled_mask

                # 最终用于模型训练的标签
                y_train = torch.zeros_like(labels)
                
                # 1. 处理无标签样本 (Stochastic EM / Hard EM)
                if unlabeled_mask.any():
                    # E-step：计算后验概率
                    with torch.no_grad():
                        unlabeled_probs = posterior_probs(
                            model,
                            z_t[unlabeled_mask],
                            z_tm1[unlabeled_mask],
                            t[unlabeled_mask],
                            K,
                            eps_eff[unlabeled_mask],
                            tau=0.5 # tau 是一个可以调整的超参数
                        )
                    
                    # M-step (Hard): 从后验分布中采样伪标签
                    # torch.multinomial 从每行的概率分布中采样一个索引
                    pseudo_labels = torch.multinomial(unlabeled_probs, num_samples=1).squeeze(1)
                    y_train[unlabeled_mask] = pseudo_labels

                # 2. 处理有标签样本
                if labeled_mask.any():
                    y_train[labeled_mask] = labels[labeled_mask]

                # 3. 统一进行模型优化
                # 现在 y_train 包含了真实标签和伪标签
                eps_pred = model(z_t, t, y_train)
                mse_loss = loss_fn(eps_pred, eps_eff).mean() # 使用 .mean() 对整个批次的损失求平均

                with torch.no_grad():
                    # 在这里使用完整的批次来计算熵
                    all_probs = posterior_probs(model, z_t, z_tm1, t, K, eps_eff, tau=0.5)
                    entropy = -(all_probs * (all_probs.clamp_min(1e-12).log())).sum(dim=1).mean().item()
                    entropy_list.append(entropy)
                    # 计算 MSE 损失
                    mse_loss = F.mse_loss(eps_pred, eps_eff, reduction='none').mean()
                    mse_loss_list.append(mse_loss.item())

                loss = mse_loss + lambda_val * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                
                

                if iter_count % 1000 == 0:
                    torch.save(model.state_dict(), model_path)
                    print(f"epoch {epoch}, iter {iter_count}, loss {total_loss.item():.4f}, posterior entropy {entropy:.3f}")
                iter_count += 1
                
        save_loss_plot(loss_list, loss_plot_name)
        save_loss_plot(mse_loss_list, loss_plot_name.replace('loss', 'mse_loss'))
        save_loss_plot(entropy_list, loss_plot_name.replace('loss', 'entropy'))
