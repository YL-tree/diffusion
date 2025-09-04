from torch.utils.data import DataLoader, random_split
from mnist_data import MNIST
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import T, forward_add_noise_pair, posterior_probs, posterior_logit

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

def calculate_differentiable_logits(model, z_t, z_tm1, t, K):
    """
    一个可微分的函数，用于计算分类损失所需的 logits。
    去掉了 @torch.no_grad() 装饰器。
    """
    device = z_t.device
    logits_list = []
    for k in range(K):
        yk = torch.full((z_t.size(0),), k, dtype=torch.long, device=device)
        
        # 梯度将从这里流过
        eps_pred_k = model(z_t, t, yk)
        
        # 调用核心的 logit 计算
        logit_k = posterior_logit(z_t, z_tm1, t, eps_pred_k)
        logits_list.append(logit_k)

    logits = torch.stack(logits_list, dim=1)  # [BatchSize, K]
    return logits

# ======================
# 验证函数 (新增)
# ======================
def validate_model(model, val_loader, K):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            x0 = imgs.to(DEVICE) * 2 - 1
            labels = labels.to(DEVICE)
            
            # 在验证时，我们通常选择一个固定的、较小的 t 来评估分类能力
            # 因为 t 太大，图像信息丢失太多，无法分类
            # t=1 或一个小的整数是常见的选择
            t = torch.ones(x0.size(0), device=DEVICE).long()
            
            z_t, z_tm1, eps_eff = forward_add_noise_pair(x0, t)

            # 直接计算后验概率并取最大值作为预测
            probs = posterior_probs(model, z_t, z_tm1, t, K, eps_eff)
            _, predicted_labels = torch.max(probs, 1)
            
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

# ======================
# 训练主程序
# ======================
if __name__ == '__main__':
    # 为了快速搜索，可以只训练有限的epoch
    EPOCH = 20 
    BATCH_SIZE = 200
    K = 10  # MNIST 类别数
    # 新增参数：从先验中生成标签的概率
    unlabeled_prob = 0

    # --- 数据准备 ---
    dataset = MNIST()
    # 分割训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 定义两个损失函数
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    # --- 步骤二：定义网格搜索空间 ---
    # (根据您在步骤一中观察到的损失量级来设定)
    lambda_grid = [0.02, 0.2, 2.0] 
    best_lambda = None
    best_val_accuracy = 0.0
    best_lambda = None
    
    for lambda_val in lambda_grid:
        print(f"Training with lambda = {lambda_val}")
        model_path = f'model/unet_{lambda_val}.pth'
        loss_plot_name = f'loss_{lambda_val}'
        acc_plot_name = f'acc_{lambda_val}'

        # --- 步骤三：训练模型 ---
        model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        try:
            model.load_state_dict(torch.load(model_path, weights_only=False))
        except FileNotFoundError:
            print("Model file not found, starting training from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}, starting from scratch.")
        
        iter_count = 0
        loss_list = []
        mse_loss_list = []
        entropy_list = []
            
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
                ce_loss = ce_loss_fn(logits_for_loss, labels) # ce_loss_fn 会自动处理 softmax

                # 3. 计算总损失
                loss = mse_loss + lambda_val * ce_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                mse_loss_list.append(mse_loss.item())
        
        # --- 步骤四：验证模型 ---
        val_accuracy = validate_model(model, val_loader, K)
        print(f"Validation Accuracy with lambda = {lambda_val}: {val_accuracy:.2f}%")
        
        # --- 步骤五：保存最佳模型 ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_lambda = lambda_val
            torch.save(model.state_dict(), 'model/unet_best.pth')
                
                
        save_loss_plot(loss_list, loss_plot_name)
        save_loss_plot(mse_loss_list, loss_plot_name.replace('loss', 'mse_loss'))
        save_loss_plot(entropy_list, loss_plot_name.replace('loss', 'entropy'))

    print("\n----- 网格搜索完成 -----")
    print(f"最佳 Lambda: {best_lambda}")
    print(f"对应的最高验证集准确率: {best_val_accuracy:.2f}%")