import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np

# =============================================================================
# 1. 配置参数 (Configuration)
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 1e-5
LATENT_DIM = 64  # z 的维度
NUM_CLASSES = 10  # x 的维度 (MNIST类别数)
UNLABELED_RATIO = 0.98  # 98%的数据将作为无标签数据使用

# 创建结果保存目录
if not os.path.exists('results/mixture_vae_reconstruction'):
    os.makedirs('results/mixture_vae_reconstruction')
if not os.path.exists('results/mixture_vae_generated'):
    os.makedirs('results/mixture_vae_generated')


# =============================================================================
# 2. 混合VAE 模型定义 (Mixture VAE Model Definition)
# =============================================================================
class MixtureVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super(MixtureVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- 编码器 (Inference Network) ---
        # 目标: 从 y 推断 p(x|y) 和 q(z|y)
        self.encoder_net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        # 输出离散变量 x 的 logits
        self.fc_logits_x = nn.Linear(256, num_classes)
        # 输出连续变量 z 的分布参数
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # --- 解码器 (Generative Network) ---
        # 目标: 从 x 和 z 生成 p(y|x, z)
        # 我们将 x (one-hot) 和 z 拼接起来作为输入
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )

    def encode(self, y_flat):
        h = self.encoder_net(y_flat)
        logits_x = self.fc_logits_x(h)
        probs_x = F.softmax(logits_x, dim=1)  # p(x|y)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return probs_x, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_onehot):
        # 将 z 和 x 拼接
        z_x_concat = torch.cat([z, x_onehot], dim=1)
        return self.decoder_net(z_x_concat)

    def forward(self, y):
        y_flat = y.view(-1, 784)
        probs_x, mu, log_var = self.encode(y_flat)
        z = self.reparameterize(mu, log_var)
        return probs_x, mu, log_var, z


# =============================================================================
# 3. 损失函数定义 (Loss Function)
# =============================================================================
def mvae_loss_function(y, probs_x, mu, log_var, z, model):
    # 将 y 展平
    y_flat = y.view(-1, 784)

    # --- E-Step in Loss (对x求期望) ---
    # 对于每个可能的类别 k，计算其对应的重构损失和KL散度
    # 然后用后验概率 p(x=k|y) 进行加权求和
    loss = 0
    for k in range(model.num_classes):
        # 构造 one-hot 向量
        x_k = torch.eye(model.num_classes, device=DEVICE)[k].expand(y.size(0), model.num_classes)

        # 解码得到重构图像
        y_reconstructed = model.decode(z, x_k)

        # 1. 重构损失 (Reconstruction Loss)
        recon_loss_k = F.binary_cross_entropy(y_reconstructed, y_flat, reduction='none').sum(dim=1)

        # 2. KL散度损失 (KL Divergence for z)
        # p(z) ~ N(0,I), q(z|y) ~ N(mu, sigma)
        kld_z = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        # 3. KL散度损失 (KL Divergence for x)
        # p(x) ~ Uniform, q(x|y) is the predicted prob
        # KL(q(x|y) || p(x))
        prior_x = torch.full_like(probs_x, 1.0 / model.num_classes)
        kld_x = (probs_x * (torch.log(probs_x + 1e-10) - torch.log(prior_x + 1e-10))).sum(dim=1)

        # 用后验概率 p(x=k|y) 对损失加权
        # E_{q(x|y)}[log p(y|x,z) - KL(q(z|y)||p(z))]
        # L = p(x=k|y) * (Recon_k + KLD_z) + KLD_x
        loss += probs_x[:, k] * (recon_loss_k + 0.01*kld_z) + kld_x

    return loss.mean()


# =============================================================================
# 4. 数据加载 (Data Loading)
# =============================================================================
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)

# 划分有标签和无标签数据集
n_samples = len(full_dataset)
n_labeled = int(n_samples * (1 - UNLABELED_RATIO))
n_unlabeled = n_samples - n_labeled
labeled_indices, unlabeled_indices = torch.utils.data.random_split(
    range(n_samples), [n_labeled, n_unlabeled]
)
# 这里我们为了简化，仍然使用所有数据的标签，但在损失函数中会忽略无标签数据的标签
# 一个更严格的实现会创建一个新的Dataset类
train_loader = DataLoader(dataset=full_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================================================================
# 5. 训练主程序 (Training)
# =============================================================================
if __name__ == '__main__':
    model = MixtureVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_load_path = 'model/mixture_vae_mnist.pth'

    try:
        model.load_state_dict(torch.load(model_load_path, weights_only=False))
    except FileNotFoundError:
        print("Model file not found, starting training from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}, starting from scratch.")

    print(f"开始在 {DEVICE} 上训练 Mixture VAE...")
    print(f"数据集中 {len(unlabeled_indices.indices)} 个样本将被视为无标签。")

    # 创建一个mask来识别无标签数据
    unlabeled_mask = torch.zeros(len(full_dataset), dtype=torch.bool)
    unlabeled_mask[unlabeled_indices.indices] = True

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch_idx, (y, x_true) in enumerate(train_loader):
            y, x_true = y.to(DEVICE), x_true.to(DEVICE)

            # 前向传播
            probs_x, mu, log_var, z = model(y)

            # --- 这是EM算法的核心 ---
            # 1. 对于有标签数据，我们使用真实标签
            # 2. 对于无标签数据，我们使用模型推断的后验 p(x|y)
            # 在这个统一的损失函数 mvae_loss_function 中，
            # 这一步已经通过对所有k求期望隐式地完成了，相当于一个"软EM"

            loss = mvae_loss_function(y, probs_x, mu, log_var, z, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'====> Epoch: {epoch} 平均损失: {avg_loss:.4f}')

        # --- 可视化 ---
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 在训练循环中保存生成结果
                num_per_class = 20  # 每个类别生成多少张图
                all_generated = []

                for k in range(NUM_CLASSES):
                    # 生成该簇的多个样本
                    z_sample = torch.randn(num_per_class, LATENT_DIM).to(DEVICE)
                    x_sample = torch.eye(NUM_CLASSES)[k].unsqueeze(0).repeat(num_per_class, 1).to(DEVICE)

                    generated = model.decode(z_sample, x_sample).cpu()
                    all_generated.append(generated)

                # 拼接所有类别的生成结果
                all_generated = torch.cat(all_generated, dim=0)

                save_image(
                    all_generated.view(NUM_CLASSES * num_per_class, 1, 28, 28),
                    f'results/mixture_vae_generated/generated_clusters_{epoch}.png',
                    nrow=num_per_class  # 每行 num_per_class 张，同一类在一行
                )

        # 训练完成后保存模型
        torch.save(model.state_dict(), 'model/mixture_vae_mnist.pth')
        print("训练完成，模型已保存。")