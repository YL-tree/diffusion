import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np

# --- 1. 参数设置 ---
# EM 训练参数
NUM_EM_EPOCHS = 10    # EM算法迭代次数
NUM_M_STEP_EPOCHS = 3 # 每次M-step中，模型训练的轮数

# 模型参数
LATENT_DIM = 20       # 潜变量z的维度
NUM_CLASSES = 10      # 类别数量 (MNIST有10类)

# 数据和训练参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建输出目录
if not os.path.exists('results_em_cvae'):
    os.makedirs('results_em_cvae')

# --- 2. 数据加载 ---
train_loader = DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

# 为了评估，我们也加载测试集
test_loader = DataLoader(
    datasets.MNIST('./', train=False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)

# --- 3. CVAE 模型定义 ---

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        image_size = 28 * 28
        # --- 编码器 ---
        # 输入: image + one-hot label
        self.encoder_net = nn.Sequential(
            nn.Linear(image_size + num_classes, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.encoder_fc_mu = nn.Linear(256, latent_dim)
        self.encoder_fc_logvar = nn.Linear(256, latent_dim)

        # --- 解码器 ---
        # 输入: latent z + one-hot label
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )

    def encode(self, y, x_onehot):
        # 将图像和标签拼接
        inputs = torch.cat([y.view(-1, 28*28), x_onehot], dim=1)
        h1 = self.encoder_net(inputs)
        return self.encoder_fc_mu(h1), self.encoder_fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_onehot):
        # 将潜变量和标签拼接
        inputs = torch.cat([z, x_onehot], dim=1)
        return self.decoder_net(inputs)

    def forward(self, y, x_onehot):
        mu, logvar = self.encode(y, x_onehot)
        z = self.reparameterize(mu, logvar)
        recon_y = self.decode(z, x_onehot)
        return recon_y, mu, logvar

# 损失函数: Negative ELBO
def loss_function(recon_y, y, mu, logvar):
    BCE = F.binary_cross_entropy(recon_y, y.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- 4. EM 算法实现 ---
def e_step(model, dataloader, label_priors, num_classes, device):
    """
    E-Step: 计算后验概率 p(x|y) (即 gamma)
    """
    model.eval()
    all_gamma = []
    all_true_labels = []

    print("Performing E-Step...")
    with torch.no_grad():
        for y_batch, true_labels in tqdm(dataloader):
            y_batch = y_batch.to(device)
            batch_size = y_batch.size(0)

            # 准备输入，为每个可能的类别计算ELBO
            # y_batch: [B, 1, 28, 28] -> [B, K, 1, 28, 28] -> [B*K, 1, 28, 28]
            y_tiled = y_batch.unsqueeze(1).repeat(1, num_classes, 1, 1, 1).view(-1, 1, 28, 28)

            # x_onehot: [K, K] -> [B, K, K] -> [B*K, K]
            x_onehot = torch.eye(num_classes, device=device)
            x_onehot_tiled = x_onehot.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, num_classes)

            # 通过模型计算重构和潜变量分布
            recon_flat, mu_flat, logvar_flat = model(y_tiled, x_onehot_tiled)
            
            # 计算负ELBO
            neg_elbo_flat = loss_function(recon_flat, y_tiled, mu_flat, logvar_flat)
            neg_elbo_batched = neg_elbo_flat.view(batch_size, num_classes)
            
            # ELBO = -neg_elbo
            elbo = -neg_elbo_batched

            # 计算 log p(y,x) = log p(y|x) + log p(x)
            # 我们用 ELBO(y,x) 来近似 log p(y|x)
            log_p_y_x = elbo + torch.log(label_priors).unsqueeze(0)

            # 使用 log-sum-exp 技巧计算 log p(y) 以保证数值稳定性
            log_p_y = torch.logsumexp(log_p_y_x, dim=1, keepdim=True)
            
            # 计算 log p(x|y) = log p(y,x) - log p(y)
            log_gamma = log_p_y_x - log_p_y
            
            gamma_batch = torch.exp(log_gamma)
            
            all_gamma.append(gamma_batch.cpu())
            all_true_labels.append(true_labels)

    return torch.cat(all_gamma, dim=0), torch.cat(all_true_labels, dim=0)


def m_step(model, optimizer, dataloader, gamma):
    """
    M-Step: 更新模型参数和类别先验
    """
    model.train()
    total_loss = 0
    
    # 创建一个新的 DataLoader，包含数据和对应的 gamma
    dataset_with_gamma = TensorDataset(dataloader.dataset.tensors[0], gamma)
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

            # 计算负ELBO (即损失)
            neg_elbo_flat = loss_function(recon_flat, y_tiled, mu_flat, logvar_flat)
            neg_elbo_batched = neg_elbo_flat.view(y_batch.size(0), NUM_CLASSES)

            # 计算加权损失
            weighted_loss = torch.sum(gamma_batch * neg_elbo_batched)
            
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()
            
        print(f"  M-Step Epoch Loss: {epoch_loss / len(dataloader.dataset):.4f}")

    # 更新类别先验 p(x)
    new_label_priors = torch.mean(gamma, dim=0)
    
    return new_label_priors

# --- 5. 训练主循环 ---

model = ConditionalVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化类别先验为均匀分布
label_priors = torch.ones(NUM_CLASSES, device=DEVICE) / NUM_CLASSES

# 加载整个训练集用于E-step
full_train_loader = DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)
full_train_dataset_tensors = next(iter(DataLoader(full_train_loader.dataset, batch_size=len(full_train_loader.dataset))))

for em_epoch in range(NUM_EM_EPOCHS):
    print(f"\n--- EM Epoch {em_epoch + 1}/{NUM_EM_EPOCHS} ---")
    
    # E-Step
    gamma, _ = e_step(model, full_train_loader, label_priors, NUM_CLASSES, DEVICE)
    
    # M-Step
    # 注意这里我们将整个训练数据和计算好的gamma传入
    new_label_priors = m_step(model, optimizer, TensorDataset(full_train_dataset_tensors[0], full_train_dataset_tensors[1]), gamma)
    
    # 打印先验概率的变化
    print(f"Old priors: {[f'{p:.3f}' for p in label_priors.cpu().numpy()]}")
    print(f"New priors: {[f'{p:.3f}' for p in new_label_priors.cpu().numpy()]}")
    label_priors = new_label_priors.to(DEVICE)

print("\n--- Training Finished ---")

# --- 6. 结果评估 ---

# (1) 评估聚类准确度
print("\nEvaluating clustering accuracy...")
# 使用 E-Step 函数来获取测试集的 gamma 分布
test_gamma, test_true_labels = e_step(model, test_loader, label_priors, NUM_CLASSES, DEVICE)
predicted_clusters = torch.argmax(test_gamma, dim=1).cpu().numpy()
test_true_labels = test_true_labels.numpy()

# 建立聚类索引到真实标签的映射
from scipy.stats import mode
mapping = {}
for i in range(NUM_CLASSES):
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
print(f"Clustering Accuracy on Test Set: {accuracy * 100:.2f}%")
print("Cluster to Label Mapping:", mapping)

# (2) 条件生成图像
print("\nGenerating images conditioned on inferred labels...")
with torch.no_grad():
    # 为每个类别生成 10 张图像
    num_gens_per_class = 10
    # 从标准正态分布中采样 z
    z_samples = torch.randn(num_gens_per_class * NUM_CLASSES, LATENT_DIM).to(DEVICE)
    
    # 创建要生成的标签
    gen_labels = torch.arange(NUM_CLASSES).repeat(num_gens_per_class)
    gen_x_onehot = F.one_hot(gen_labels, NUM_CLASSES).to(DEVICE).float()
    
    generated_images = model.decode(z_samples, gen_x_onehot).cpu()
    
    # 将生成的图像保存为网格图
    save_image(generated_images.view(-1, 1, 28, 28),
               'results_em_cvae/generated_samples.png',
               nrow=num_gens_per_class)

print("Generated images saved to 'results_em_cvae/generated_samples.png'")