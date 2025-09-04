from torch.utils.data import DataLoader
from mnist_data import MNIST
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import T, forward_add_noise_pair, posterior_logit

#----- 网格搜索完成 -----
#最佳 Lambda: 0.2
#对应的最高验证集准确率: 98.28%

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

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

def calculate_differentiable_logits(model, z_t, z_tm1, t, K):
    device = z_t.device
    logits_list = torch.zeros(z_t.size(0), K, device=device)
    
    # 获取 t > 1 和 t = 1 的掩码
    is_t_gt_1 = (t > 1)
    is_t_eq_1 = (t == 1)
    
    # ==========================
    # 对 t > 1 的样本进行处理
    # ==========================
    if is_t_gt_1.any():
        z_t_gt1 = z_t[is_t_gt_1]
        z_tm1_gt1 = z_tm1[is_t_gt_1]
        t_gt1 = t[is_t_gt_1]
        
        logits_gt1 = torch.zeros(z_t_gt1.size(0), K, device=device)
        for k in range(K):
            yk = torch.full((z_t_gt1.size(0),), k, dtype=torch.long, device=device)
            eps_pred_k = model(z_t_gt1, t_gt1, yk)
            logit_k = posterior_logit(z_t_gt1, z_tm1_gt1, t_gt1, eps_pred_k)
            logits_gt1[:, k] = logit_k
        
        logits_list[is_t_gt_1] = logits_gt1

    # ==========================
    # 对 t = 1 的样本进行特殊处理
    # ==========================
    if is_t_eq_1.any():
        # 这里你可以选择跳过这个样本，或者用一个小的常数/0来避免NaN。
        # 最简单粗暴的方法是：对这些样本的CE Loss贡献为0，不参与CE损失计算。
        # 也就是让它们的 logits 变为 0，这样 ce_loss_fn 在计算时就会忽略它们。
        logits_eq1 = torch.zeros(z_t[is_t_eq_1].size(0), K, device=device)
        logits_list[is_t_eq_1] = logits_eq1

    return logits_list

# ======================
# 训练主程序
# ======================
if __name__ == '__main__':
    # 为了快速搜索，可以只训练有限的epoch
    EPOCH = 100 
    BATCH_SIZE = 256
    K = 10  # MNIST 类别数
    # 新增参数：从先验中生成标签的概率
    unlabeled_prob = 0
    lambda_val = 0.2

    # --- 数据准备 ---
    dataset = MNIST()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 定义model, 损失函数, 优化器 ---
    model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
    model_path = f'model/unet_best.pth'
    try:
        model.load_state_dict(torch.load(model_path, weights_only=False))
    except FileNotFoundError:
        print("Model file not found, starting training from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}, starting from scratch.")

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    loss_plot_name = f'loss_best'
    acc_plot_name = f'acc_best'

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    iter_count = 0
    loss_list = []
    mse_loss_list = []
    entropy_list = []

    print(f"Starting training with lambda = {lambda_val}")
    # --- 训练模型 --- 
    for epoch in tqdm(range(EPOCH), desc='Training', unit='epoch'):
        model.train()
        for imgs, labels in train_loader:
            x0 = imgs.to(DEVICE) * 2 - 1
            labels = labels.to(DEVICE)
            t = torch.randint(1, T + 1, (x0.size(0),), device=DEVICE).long()
            
            z_t, z_tm1, eps_eff = forward_add_noise_pair(x0, t)

            eps_pred = model(z_t, t, labels)

            # 1. 计算 MSE 损失
            mse_loss = mse_loss_fn(eps_pred, eps_eff)

            # 2. 计算 CE 损失
            logits_for_loss = calculate_differentiable_logits(model, z_t, z_tm1, t, K)
            ce_loss = ce_loss_fn(logits_for_loss, labels) 

            # 3. 计算总损失
            loss = mse_loss + lambda_val * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            mse_loss_list.append(mse_loss.item())
            entropy_list.append(ce_loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCH}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}")
            torch.save(model.state_dict(), model_path)
            save_loss_plot(loss_list, loss_plot_name)
            save_loss_plot(mse_loss_list, loss_plot_name.replace('loss', 'mse_loss'))
            save_loss_plot(entropy_list, loss_plot_name.replace('loss', 'entropy'))

    print("Training finished. Model and plots saved.")
    
    
    
            
    