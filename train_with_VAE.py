import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.vision import data
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from torch.utils.data import random_split

# --- 1. 参数设置 ---
# EM 训练参数
NUM_EM_EPOCHS = 50   # EM算法迭代次数
NUM_M_STEP_EPOCHS = 1 # 每次M-step中，模型训练的轮数

# 模型参数
LATENT_DIM = 18       # 潜变量z的维度
NUM_CLASSES = 10      # 类别数量 (MNIST有10类)

# 数据和训练参数
BATCH_SIZE = 100
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_INTERVAL = 2 
SAVE_PATH = 'results_em_cvae_test'

# 创建输出目录
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(os.path.join(SAVE_PATH, 'loss')):
    os.makedirs(os.path.join(SAVE_PATH, 'loss'))
if not os.path.exists(os.path.join(SAVE_PATH, 'model')):
    os.makedirs(os.path.join(SAVE_PATH, 'model'))
if not os.path.exists(os.path.join(SAVE_PATH, 'samples')):
    os.makedirs(os.path.join(SAVE_PATH, 'samples'))


# --- 2. 数据加载 ---
full_train_loader = DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)
full_train_dataset_tensors = next(iter(DataLoader(full_train_loader.dataset, batch_size=len(full_train_loader.dataset))))

# 获取完整的训练数据集
train_images, train_labels = full_train_dataset_tensors
train_images = train_images.to(DEVICE)

# 将测试集划分为验证集和测试集
data_loader = DataLoader(
    datasets.MNIST('./', train=False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)

# 计算验证集和测试集的大小
total_size = len(data_loader.dataset)
val_size = int(total_size * 0.5)
test_size = total_size - val_size
# 随机划分
val_dataset, test_dataset = random_split(data_loader.dataset, [val_size, test_size])
# 验证集和测试集的DataLoader
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 3. CVAE 模型定义 ---
class ConditionalVAE_CNN(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConditionalVAE_CNN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- 编码器 ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # -> (N, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (N, 32, 7, 7)
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(64 * 7 * 7 + num_classes, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # --- 解码器 ---
        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.decoder_fc2 = nn.Linear(256, 64 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # -> (N, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2), # -> (N, 1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, y, x_onehot):
        h_conv = self.encoder_conv(y)                 # (N, 32, 7, 7)
        h_conv_flat = h_conv.view(h_conv.size(0), -1)
        inputs = torch.cat([h_conv_flat, x_onehot], dim=1)
        h_fc = F.relu(self.encoder_fc(inputs))
        return self.fc_mu(h_fc), self.fc_logvar(h_fc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_onehot):
        inputs = torch.cat([z, x_onehot], dim=1)
        h_fc1 = F.relu(self.decoder_fc1(inputs))
        h_fc2 = F.relu(self.decoder_fc2(h_fc1))
        h_deconv_input = h_fc2.view(h_fc2.size(0), 64, 7, 7)
        return self.decoder_deconv(h_deconv_input)

    def forward(self, y, x_onehot):
        mu, logvar = self.encode(y, x_onehot)
        z = self.reparameterize(mu, logvar)
        recon_y = self.decode(z, x_onehot)
        return recon_y, mu, logvar

# 损失函数: Negative ELBO
def loss_function(recon_y, y, mu, logvar):
    # recon_y: (N, 1, 28, 28), y: (N, 1, 28, 28)
    BCE = F.binary_cross_entropy(recon_y, y, reduction='none').sum(dim=[1, 2, 3])
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return BCE + KLD

# --- 4. EM 算法实现 ---
def e_step(model, dataloader, label_priors, num_classes, device):
    """
    E-Step: 计算后验概率 p(x|y) (即 gamma)
    """
    model.eval()
    all_gamma = []
    all_true_labels = []

    print("Performing E-Step (using reconstruction loss only)...") # 提示用户当前模式
    with torch.no_grad():
        for y_batch, true_labels in tqdm(dataloader):
            y_batch = y_batch.to(device)
            batch_size = y_batch.size(0)

            # 准备输入，为每个可能的类别计算重构损失
            # y_batch: [B, 1, 28, 28] -> [B, K, 1, 28, 28] -> [B*K, 1, 28, 28]
            # 重复图像 B 次，每次搭配一个类别
            y_tiled = y_batch.unsqueeze(1).repeat(1, num_classes, 1, 1, 1)  # (B, K, 1, 28, 28)
            y_tiled = y_tiled.view(-1, 1, 28, 28)

            # x_onehot: [K, K] -> [B, K, K] -> [B*K, K]
            x_onehot = torch.eye(num_classes, device=device)
            x_onehot_tiled = x_onehot.unsqueeze(0).repeat(batch_size, 1, 1)
            x_onehot_tiled = x_onehot_tiled.view(-1, num_classes)

            # 通过模型计算重构和潜变量分布
            # 注意：即使我们只用重构损失，也需要通过完整的模型来获得重构结果
            recon_flat, _, _ = model(y_tiled, x_onehot_tiled)
            
            # ======================= 这里是核心修改 =======================
            # 原代码:
            # neg_elbo_flat = loss_function(recon_flat, y_tiled, mu_flat, logvar_flat)
            # neg_elbo_batched = neg_elbo_flat.view(batch_size, num_classes)
            # elbo = -neg_elbo_batched

            # 新代码:
            # 只计算重构损失 (BCE), 它是负对数似然 E[log p(y|z,x)] 的近似
            # 我们需要每个样本的损失，所以 reduction='none'
            recon_loss_flat = F.binary_cross_entropy(recon_flat, y_tiled, reduction='none').sum(dim=[1, 2, 3])
            
            # 将损失 reshape 成 [batch_size, num_classes]
            recon_loss_batched = recon_loss_flat.view(batch_size, num_classes)
            
            # 对数似然 log p(y|x) 就用 -recon_loss 来近似
            log_likelihood = -recon_loss_batched
            # =============================================================
            
            # 计算 log p(y,x) = log p(y|x) + log p(x)
            # 我们用近似的 log_likelihood 来代表 log p(y|x)
            log_p_y_x = log_likelihood + torch.log(label_priors).unsqueeze(0)

            

            # # 使用温度参数进行锐化
            # temperature = 0.5 # 超参数，可以从接近1开始慢慢减小，例如0.9, 0.8...
            # log_p_y_x_sharpened = log_p_y_x / temperature
            # # 使用 log-sum-exp 技巧 计算 log p(y) 以保证数值稳定性
            # log_p_y = torch.logsumexp(log_p_y_x_sharpened, dim=1, keepdim=True)
            # log_gamma = log_p_y_x_sharpened - log_p_y

            # 使用 log-sum-exp 技巧
            log_p_y = torch.logsumexp(log_p_y_x, dim=1, keepdim=True)
            # 计算 log p(x|y) = log p(y,x) - log p(y)
            log_gamma = log_p_y_x - log_p_y
            gamma_batch = torch.exp(log_gamma)
            all_gamma.append(gamma_batch.cpu())
            all_true_labels.append(true_labels)

    return torch.cat(all_gamma, dim=0), torch.cat(all_true_labels, dim=0)

def m_step(model, optimizer, dataloader, gamma, beta=1.0):
    """
    M-Step: 更新模型参数和类别先验
    """
    model.train()
    # 创建一个新的 DataLoader，包含数据和对应的 gamma
    dataset_with_gamma = TensorDataset(dataloader.tensors[0], gamma)
    loader_with_gamma = DataLoader(dataset_with_gamma, batch_size=BATCH_SIZE, shuffle=True)

    print("Performing M-Step...")
    for _ in range(NUM_M_STEP_EPOCHS):
        epoch_loss = 0
        for y_batch, gamma_batch in tqdm(loader_with_gamma):
            y_batch, gamma_batch = y_batch.to(DEVICE), gamma_batch.to(DEVICE)
            optimizer.zero_grad()
            # 同样地，准备输入以计算所有类别的损失    
            y_tiled = y_batch.unsqueeze(1).repeat(1, NUM_CLASSES, 1, 1, 1).view(-1, 1, 28, 28)
            x_onehot = torch.eye(NUM_CLASSES, device=DEVICE)
            x_onehot_tiled = x_onehot.unsqueeze(0).repeat(y_batch.size(0), 1, 1).view(-1, NUM_CLASSES)
            
            recon_flat, mu_flat, logvar_flat = model(y_tiled, x_onehot_tiled)

            # 分别计算重构损失和KL散度
            recon_loss_part = F.binary_cross_entropy(recon_flat, y_tiled, reduction='none').sum(dim=[1, 2, 3])
            kld_part = -0.5 * torch.sum(1 + logvar_flat - mu_flat.pow(2) - logvar_flat.exp(), dim=1)

            # 使用 beta 加权 KL 散度
            neg_elbo_flat = recon_loss_part + beta * kld_part
            # 计算负ELBO (即损失)
            # neg_elbo_flat = loss_function(recon_flat, y_tiled, mu_flat, logvar_flat)
            neg_elbo_batched = neg_elbo_flat.view(y_batch.size(0), NUM_CLASSES)

            # 计算加权损失
            weighted_loss = torch.sum(gamma_batch * neg_elbo_batched)
            
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()
            
        print(f"  M-Step Epoch Loss: {epoch_loss / len(dataloader):.4f}")

    # 更新类别先验 p(x)
    new_label_priors = torch.mean(gamma, dim=0)
    
    return new_label_priors, epoch_loss / len(dataloader)

def m_step_sampled(model, optimizer, dataloader, gamma, beta=1.0):
    """
    M-Step (Sampled EM): 从 gamma 分布中采样类别作为伪标签，
    并统计类别计数来更新先验。
    """
    model.train()

    # 统计每个类别的计数
    class_counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    # === 先采样标签 ===
    sampled_labels = torch.multinomial(gamma, num_samples=1).squeeze(1).to(DEVICE)   # (N,)
    # 直接做 bincount 来统计类别分布
    class_counts += torch.bincount(sampled_labels, minlength=NUM_CLASSES).float()
    new_label_priors = class_counts / class_counts.sum()  # 归一化

    # === 构建伪标签 one-hot ===
    sampled_onehot = F.one_hot(sampled_labels, NUM_CLASSES).float()
    dataset_with_labels = TensorDataset(dataloader.tensors[0], sampled_onehot)
    loader_with_labels = DataLoader(dataset_with_labels, batch_size=BATCH_SIZE, shuffle=True)

    print("Performing Sampled M-Step...")
    total_loss = 0
    num_batches = 0

    for _ in range(NUM_M_STEP_EPOCHS):
        epoch_loss = 0
        for y_batch, sampled_onehot in tqdm(loader_with_labels):
            y_batch, sampled_onehot = y_batch.to(DEVICE), sampled_onehot.to(DEVICE)  # (B, 1, 28, 28), (B, K)
            optimizer.zero_grad()

            # 前向传播
            recon, mu, logvar = model(y_batch, sampled_onehot)

            # VAE 损失
            recon_loss_part = F.binary_cross_entropy(recon, y_batch, reduction='none').sum(dim=[1, 2, 3])
            kld_part = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # 使用 beta 加权 KL 散度
            loss = (recon_loss_part + beta * kld_part).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1

        print(f"  Sampled M-Step Epoch Loss: {epoch_loss / len(loader_with_labels):.4f}")

    avg_loss = total_loss / num_batches
    return new_label_priors, avg_loss


# -----------------
# 1. 在脚本开头定义超参数
# -----------------
# 对比损失的权重。这是一个关键的超参数，需要调试。
# 建议从 0.1, 0.5, 1.0 开始尝试。
lambda_contrast = 0.5

def m_step_contrastive(model, optimizer, dataloader, gamma, beta=1.0):
    """
    M-Step with Contrastive Learning for VAE.
    在潜空间的均值向量 mu 上施加斥力。
    """
    model.train()

    # --- 这部分与您之前的逻辑完全相同：采样标签、计算新先验、创建DataLoader ---
    class_counts = torch.zeros(NUM_CLASSES, device=DEVICE)
    sampled_labels = torch.multinomial(gamma, num_samples=1).squeeze(1).to(DEVICE)
    class_counts += torch.bincount(sampled_labels, minlength=NUM_CLASSES).float()
    new_label_priors = class_counts / class_counts.sum()

    sampled_onehot = F.one_hot(sampled_labels, NUM_CLASSES).float()
    dataset_with_labels = TensorDataset(dataloader.tensors[0], sampled_onehot)
    loader_with_labels = DataLoader(dataset_with_labels, batch_size=BATCH_SIZE, shuffle=True)
    # --- 采样逻辑结束 ---

    print("Performing Contrastive M-Step for VAE...")
    total_loss = 0
    num_batches = 0

    for _ in range(NUM_M_STEP_EPOCHS):
        epoch_loss = 0
        for y_batch, x_correct_onehot in tqdm(loader_with_labels):
            y_batch, x_correct_onehot = y_batch.to(DEVICE), x_correct_onehot.to(DEVICE)
            batch_size = y_batch.size(0)
            optimizer.zero_grad()
            
            # ======================= 对比学习核心修改 =======================
            
            # 1. 生成“错误”标签 x_wrong
            x_correct_indices = torch.argmax(x_correct_onehot, dim=1)
            rand_offset = torch.randint(1, NUM_CLASSES, (batch_size,), device=DEVICE)
            x_wrong_indices = (x_correct_indices + rand_offset) % NUM_CLASSES
            x_wrong_onehot = F.one_hot(x_wrong_indices, NUM_CLASSES).float()

            # 2. 高效地进行两次编码
            # 我们将 correct 和 wrong 的输入拼接，只调用一次编码器
            y_combined = torch.cat([y_batch, y_batch], dim=0)
            x_combined = torch.cat([x_correct_onehot, x_wrong_onehot], dim=0)
            
            # 编码器输出两倍批次大小的 mu 和 logvar
            mu_combined, logvar_combined = model.encode(y_combined, x_combined)
            
            # 拆分回 correct 和 wrong 两部分
            mu_correct, mu_wrong = torch.chunk(mu_combined, 2, dim=0)
            logvar_correct, _ = torch.chunk(logvar_combined, 2, dim=0) # logvar_wrong 不需要

            # 3. 计算“引力”部分：标准的 VAE 损失
            # 这部分需要完整的重构过程
            z_correct = model.reparameterize(mu_correct, logvar_correct)
            recon_correct = model.decode(z_correct, x_correct_onehot)
            
            recon_loss = F.binary_cross_entropy(recon_correct, y_batch, reduction='none').sum(dim=[1, 2, 3])
            kld_loss = -0.5 * torch.sum(1 + logvar_correct - mu_correct.pow(2) - logvar_correct.exp(), dim=1)
            
            standard_loss = (recon_loss + beta * kld_loss).mean()

            # 4. 计算“斥力”部分：对比损失
            # 我们希望 mu_correct 和 mu_wrong 尽可能远
            # 最大化 ||mu_correct - mu_wrong||^2 等价于最小化 -||mu_correct - mu_wrong||^2
            contrastive_loss = -F.mse_loss(mu_correct, mu_wrong)

            # 5. 合并总损失
            total_loss_batch = standard_loss + lambda_contrast * contrastive_loss
            
            # ======================= 修改结束 =======================

            total_loss_batch.backward()
            optimizer.step()

            epoch_loss += total_loss_batch.item()
            num_batches += 1
            
        print(f"  Contrastive M-Step Epoch Loss: {epoch_loss / len(loader_with_labels):.4f}")
        
    avg_loss = (total_loss + epoch_loss) / num_batches  # 修正avg_loss的计算
    return new_label_priors, avg_loss



# --- 6. 结果评估 ---
def validation(model, epoch, test_loader, label_priors, num_classes, device, latent_dim, save_dir=SAVE_PATH):
    model.eval()
    with torch.no_grad():
        # (1) 评估聚类准确度
        # 使用 E-Step 函数来获取测试集的 gamma 分布
        test_gamma, test_true_labels = e_step(model, test_loader, label_priors, num_classes, device)
        predicted_clusters = torch.argmax(test_gamma, dim=1).cpu().numpy()
        test_true_labels = test_true_labels.numpy()

        # 建立聚类索引到真实标签的映射
        mapping = {}
        for i in range(num_classes):
            # 找到所有被预测为聚类 i 的真实标签
            true_labels_for_cluster_i = test_true_labels[predicted_clusters == i]
            if len(true_labels_for_cluster_i) > 0:
                # 将这个聚类映射到最常见的真实标签
                most_common_label = mode(true_labels_for_cluster_i, keepdims=False)[0]
                mapping[i] = most_common_label

        # 计算准确率
        correct_predictions = 0
        for i in range(len(predicted_clusters)):
            if predicted_clusters[i] in mapping:
                if mapping[predicted_clusters[i]] == test_true_labels[i]:
                    correct_predictions += 1

        accuracy = correct_predictions / len(predicted_clusters)
        print(f"[Validation] Clustering Accuracy on Test Set: {accuracy * 100:.2f}%")
        print("Cluster to Label Mapping:", mapping)

        # (2) 条件生成图像
        print("\n[Validation] Generating images conditioned on inferred labels...")
        with torch.no_grad():
            # 为每个类别生成 10 张图像
            num_gens_per_class = 10
            # 从标准正态分布中采样 z
            z_samples = torch.randn(num_gens_per_class * num_classes, latent_dim).to(device)
            
            # 创建要生成的标签
            gen_labels = torch.arange(num_classes).repeat(num_gens_per_class)
            gen_x_onehot = F.one_hot(gen_labels, num_classes).to(device).float()
            
            generated_images = model.decode(z_samples, gen_x_onehot).cpu()
            save_path = os.path.join(save_dir, f"samples/generated_samples_epoch_{epoch}.png")  
            save_image(generated_images,
               save_path,
               nrow=num_gens_per_class)
               
        print(f"[Validation] Generated images saved to {save_path}")

        return accuracy, mapping

def loss_plot(loss_history, save_path=os.path.join(SAVE_PATH, "loss_plot.png")):
    """
    绘制训练损失的图表。
    """
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(loss_history)), loss_history, marker='o', linestyle='-')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


# --- 5. 训练主循环 ---
# # load model
# sup_load_path = 'results_em_cvae_supervised'
# model_path = os.path.join(sup_load_path, 'model/cvae_mnist.pth')
# model = ConditionalVAE_CNN(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
# try:
#     model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEVICE))
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()
    
model = ConditionalVAE_CNN(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化类别先验为均匀分布
label_priors = torch.ones(NUM_CLASSES, device=DEVICE) / NUM_CLASSES


# 记录loss
loss_history = []
# beta 从0线性增长到1，假设在前50个EM epoch完成
beta_anneal_epochs = 20 


for em_epoch in range(NUM_EM_EPOCHS):
    print(f"\n--- EM Epoch {em_epoch + 1}/{NUM_EM_EPOCHS} ---")

    # 计算当前epoch的beta值
    current_beta = 1.0
    # current_beta = min(1.0, em_epoch / beta_anneal_epochs)
    # print(f"Current Beta: {current_beta:.4f}")
    # E-Step
    gamma, _ = e_step(model, full_train_loader, label_priors, NUM_CLASSES, DEVICE)
    
    # M-Step
    # 注意这里我们将整个训练数据和计算好的gamma传入
    # new_label_priors, epoch_loss = m_step(model, optimizer, TensorDataset(full_train_dataset_tensors[0], full_train_dataset_tensors[1]), gamma, current_beta)
    # new_label_priors, epoch_loss = m_step_sampled(model, optimizer, 
    #                                               TensorDataset(full_train_dataset_tensors[0],
    #                                                             full_train_dataset_tensors[1]), 
    #                                               gamma, 
    #                                               current_beta)
    new_label_priors, epoch_loss = m_step_contrastive(model, optimizer, 
                                                  TensorDataset(full_train_dataset_tensors[0],
                                                                full_train_dataset_tensors[1]), 
                                                  gamma, 
                                                  current_beta)
                                                  
    loss_history.append(epoch_loss)

    if (em_epoch + 1) % 10 == 0:
        # 打印先验概率的变化
        print(f"Old priors: {[f'{p:.3f}' for p in label_priors.cpu().numpy()]}")
        print(f"New priors: {[f'{p:.3f}' for p in new_label_priors.cpu().numpy()]}")
    label_priors = new_label_priors.to(DEVICE)

    # 验证
    if (em_epoch + 1) % VALIDATION_INTERVAL == 0:
        validation(model, em_epoch + 1, val_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
        # 绘制loss图表
        loss_plot(loss_history, os.path.join(SAVE_PATH, f"loss/loss_plot_epoch_{em_epoch + 1}.png"))


print("\n--- Training Finished ---")
# 保存模型
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model/cvae_mnist.pth'))
print("Model saved to model/cvae_mnist.pth")

# 保存loss历史
np.save(os.path.join(SAVE_PATH, 'model/cvae_mnist_loss_history.npy'), loss_history)
print("Loss history saved to model/cvae_mnist_loss_history.npy")

# 测试集验证
test_accuracy, _ = validation(model, NUM_EM_EPOCHS, test_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
print(f"[Test] Clustering Accuracy on Test Set: {test_accuracy * 100:.2f}%")


