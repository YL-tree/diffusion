import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt

def save_loss_plot(loss_list, plot_name, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label=f'Training {plot_name}')
    plt.xlabel('Iterations')
    plt.ylabel(f'{plot_name}')
    plt.title(f'Training {plot_name} Over Time')
    plt.legend()
    plt.savefig(f'results/mixture_vae_3_{plot_name}.png')
    plt.close()

# =============================================================================
# 1. 配置参数 (Configuration)
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100
EPOCHS = 100
# --- MODIFIED: 调整了学习率到一个更常见的值 ---
LEARNING_RATE = 1e-4
LATENT_DIM = 64  # z 的维度
NUM_CLASSES = 10  # x 的维度 (MNIST类别数)
UNLABELED_RATIO = 0.98  # 98%的数据将作为无标签数据使用

# 创建结果保存目录
if not os.path.exists('results/mixture_vae_reconstruction_4'):
    os.makedirs('results/mixture_vae_reconstruction_4')
if not os.path.exists('results/mixture_vae_generated_4'):
    os.makedirs('results/mixture_vae_generated_4')
if not os.path.exists('model/VAE_4'):
    os.makedirs('model/VAE_4')

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
        # 增强 x_onehot 的信号
        x_enhanced = x_onehot * 10.0 # 10.0 是一个可以调整的超参数
        
        # 将 z 和增强后的 x 拼接
        z_x_concat = torch.cat([z, x_enhanced], dim=1)
        return self.decoder_net(z_x_concat)

    def forward(self, y):
        y_flat = y.view(-1, 784)
        probs_x, mu, log_var = self.encode(y_flat)
        z = self.reparameterize(mu, log_var)
        return probs_x, mu, log_var, z


# =============================================================================
# 3. 损失函数定义 (Loss Function)
# =============================================================================
# --- MODIFIED: 完全重写了损失函数以修正逻辑错误 ---
def mvae_loss_function(y, probs_x, mu, log_var, z, model):
    """
    计算混合VAE的损失函数 (-ELBO)。
    -ELBO = E_{q(x|y)}[log p(y|x,z)] + E_{q(x|y)}[KL(q(z|y)||p(z))] + KL(q(x|y)||p(x))
    在实践中，我们最小化：
    重构损失的期望 + beta * KL散度(z) + alpha * KL散度(x)
    """
    y_flat = y.view(-1, 784)

    # --- 第一部分: 重构损失的期望 E_{q(x|y)}[log p(y|x,z)] ---
    # 通过对所有可能的类别k进行加权求和来计算期望
    expected_recon_loss = 0
    for k in range(model.num_classes):
        # 为类别k创建one-hot向量
        x_k = torch.eye(model.num_classes, device=DEVICE)[k].expand(y.size(0), model.num_classes)
        
        # 使用类别k的one-hot向量和z进行解码，得到重构图像
        y_reconstructed = model.decode(z, x_k)
        
        # 计算类别k下的重构损失 (负对数似然)
        recon_loss_k = F.binary_cross_entropy(y_reconstructed, y_flat, reduction='none').sum(dim=1)
        
        # 使用后验概率 p(x=k|y) (即 probs_x[:, k]) 进行加权
        expected_recon_loss += probs_x[:, k] * recon_loss_k

    # --- 第二部分: 连续潜变量z的KL散度 ---
    # KL(q(z|y) || p(z)) where p(z) is N(0,I)
    kld_z = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

    # --- 第三部分: 离散潜变量x的KL散度 ---
    # KL(q(x|y) || p(x)) where p(x) is a uniform categorical distribution
    prior_x = torch.full_like(probs_x, 1.0 / model.num_classes)
    # 添加一个小的epsilon防止log(0)
    kld_x = (probs_x * (torch.log(probs_x + 1e-10) - torch.log(prior_x + 1e-10))).sum(dim=1)

    # --- 组合所有部分形成最终的损失 (-ELBO) ---
    # 我们加上KL散度，因为我们是在最小化(-ELBO)，而KL散度是正的
    # 这里的 0.01 是一个超参数(beta)，用于平衡重构和正则化
    alpha = 0.2
    beta = 0.5
    loss = expected_recon_loss + beta * kld_z + alpha * kld_x

    return loss.mean(), expected_recon_loss.mean(), kld_z.mean(), kld_x.mean()


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
    model_load_path = 'model/VAE_4/mixture_vae_mnist.pth'

    try:
        model.load_state_dict(torch.load(model_load_path, weights_only=False))
        print(f"成功从 {model_load_path} 加载模型。")
    except FileNotFoundError:
        print("模型文件未找到，将从头开始训练。")
    except Exception as e:
        print(f"加载模型时出错: {e}，将从头开始训练。")

    print(f"开始在 {DEVICE} 上训练 Mixture VAE...")
    print(f"数据集中 {len(unlabeled_indices.indices)} 个样本将被视为无标签。")
    
    # 记录损失
    loss_list = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch_idx, (y, x_true) in enumerate(train_loader):
            y, x_true = y.to(DEVICE), x_true.to(DEVICE)

            # 前向传播
            probs_x, mu, log_var, z = model(y)

            # --- 这是EM算法的核心 ---
            # 当前的损失函数是一个纯粹的无监督实现 ("软EM")
            # 它对所有数据都使用模型推断的后验 p(x|y)
            #
            # (可选) 如果要实现半监督学习:
            # 你需要一个mask来区分有标签和无标签的样本
            # 1. 对无标签样本，使用当前的 mvae_loss_function
            # 2. 对有标签样本，构造一个不同的损失：
            #    - 重构损失: 只计算真实类别 x_true 对应的重构
            #    - 分类损失: 添加一个交叉熵损失，鼓励 probs_x 接近 x_true 的 one-hot 编码

            loss, expected_recon_loss, kld_z, kld_x = mvae_loss_function(y, probs_x, mu, log_var, z, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                loss_list.append(loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f'====> Epoch: {epoch} 平均损失: {avg_loss:.4f} 重构损失: {expected_recon_loss:.4f} KL(z): {kld_z:.4f} KL(x): {kld_x:.4f} ')       

        # --- 可视化 ---
        if epoch % 10 == 1 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                # 可视化重构结果
                # 从测试集中取一些样本
                test_dataset = datasets.MNIST(root='./', train=False, transform=transform, download=True)
                test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
                test_iter = iter(test_loader)
                y_test, x_true_test = next(test_iter)  # Get test labels
                y_test = y_test.to(DEVICE)
                x_true_test = x_true_test.to(DEVICE)

                # 编码并找到概率最高的类别
                probs_x_test, _, _, z_test = model(y_test)
                x_pred = torch.argmax(probs_x_test, dim=1)
                x_onehot_pred = F.one_hot(x_pred, num_classes=NUM_CLASSES).float()
                

                # 使用预测的类别进行解码重构
                y_reconstructed_test = model.decode(z_test, x_onehot_pred)

                # 保存对比图像
                comparison = torch.cat([y_test.view(-1, 1, 28, 28), 
                                        y_reconstructed_test.view(-1, 1, 28, 28)])
                save_image(comparison.cpu(),
                           f'results/mixture_vae_reconstruction_4/reconstruction_epoch_{epoch}.png', nrow=y_test.size(0))


                # 可视化按类别生成的结果
                num_per_class = 10  # 每个类别生成多少张图
                all_generated = []

                for k in range(NUM_CLASSES):
                    z_sample = torch.randn(num_per_class, LATENT_DIM).to(DEVICE)
                    x_sample = torch.eye(NUM_CLASSES)[k].unsqueeze(0).repeat(num_per_class, 1).to(DEVICE)
                    generated = model.decode(z_sample, x_sample).cpu()
                    all_generated.append(generated)

                all_generated = torch.cat(all_generated, dim=0)
                save_image(
                    all_generated.view(NUM_CLASSES * num_per_class, 1, 28, 28),
                    f'results/mixture_vae_generated_4/generated_clusters_epoch_{epoch}.png',
                    nrow=num_per_class
                )
                torch.save(model.state_dict(), f"model/VAE_4/mixture_vae_mnist_{epoch}.pth")
                # 绘制损失图
                save_loss_plot(loss_list, 'loss', epoch)

    # 训练完成后保存模型
    torch.save(model.state_dict(), model_load_path)
    print(f"训练完成，模型已保存至 {model_load_path}。")