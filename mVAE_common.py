# mVAE_common.py
# ★★ v4: 论文公式从 epoch 1 开始, 不分阶段
#   添加两个正则化 (不改变 ELBO 结构):
#     1. z_dropout: decoder 正则化, 迫使 decoder 依赖 x
#     2. balance_loss: 等价于 Π 的 Dirichlet 先验, 防止类坍缩

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

# ============================================================
# Config
# ============================================================
class Config:
    latent_dim = 2
    hidden_dim = 256
    num_classes = 10

    beta = 2.0
    use_bce = True

    # Gumbel
    init_gumbel_temp = 1.0
    min_gumbel_temp = 0.1
    gumbel_anneal_rate = 0.98
    current_gumbel_temp = init_gumbel_temp

    # ★★ v4: 两个正则化 (论文框架兼容)
    z_dropout_rate = 0.5           # decoder 侧 z dropout
    balance_weight = 10.0          # balance loss 权重

    # 训练
    lr = 1e-3
    batch_size = 128
    optuna_epochs = 15
    final_epochs = 100
    decoder_type = 'film'

    # 半监督
    alpha_unlabeled = 0.5
    labeled_per_class = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "./mVAE_results"


# ============================================================
# Encoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)


# ============================================================
# Conditional Decoder (Concatenation)
# ============================================================
class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim=16, num_classes=10, hidden_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        return self.decoder(self.fc(h))


# ============================================================
# FiLM Decoder
# ============================================================
class FiLMDecoder(nn.Module):
    def __init__(self, latent_dim=4, num_classes=10, hidden_dim=256):
        super().__init__()
        self.class_embed = nn.Sequential(
            nn.Linear(num_classes, 64), nn.ReLU(), nn.Linear(64, 64))
        self.z_fc = nn.Linear(latent_dim, hidden_dim)
        self.film1_gamma = nn.Linear(64, hidden_dim)
        self.film1_beta = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64 * 7 * 7)
        self.film2_gamma = nn.Linear(64, 64)
        self.film2_beta = nn.Linear(64, 64)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU())
        self.film3_gamma = nn.Linear(64, 32)
        self.film3_beta = nn.Linear(64, 32)
        self.final_conv = nn.Sequential(nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, z, y_onehot):
        c = self.class_embed(y_onehot)
        h = F.relu(self.z_fc(z))
        h = self.film1_gamma(c) * h + self.film1_beta(c)
        h = F.relu(h)
        h = F.relu(self.fc2(h)).view(-1, 64, 7, 7)
        g2 = self.film2_gamma(c).unsqueeze(-1).unsqueeze(-1)
        b2 = self.film2_beta(c).unsqueeze(-1).unsqueeze(-1)
        h = F.relu(g2 * h + b2)
        h = self.deconv(h)
        g3 = self.film3_gamma(c).unsqueeze(-1).unsqueeze(-1)
        b3 = self.film3_beta(c).unsqueeze(-1).unsqueeze(-1)
        h = F.relu(g3 * h + b3)
        return self.final_conv(h)


# ============================================================
# 工具函数
# ============================================================
def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

def gumbel_softmax_sample(logits, temperature, hard=False):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    y_soft = F.softmax((logits + gumbel) / (temperature + 1e-9), dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft

def compute_recon_loglik(dec, z, y, K, use_bce=True):
    B = z.size(0)
    device = z.device
    recon_loglik = []
    recon_images = []
    for k in range(K):
        y_onehot = F.one_hot(torch.full((B,), k, device=device, dtype=torch.long),
                             num_classes=K).float()
        x_recon = dec(z, y_onehot)
        recon_images.append(x_recon)
        if use_bce:
            log_p = -F.binary_cross_entropy(x_recon, y, reduction='none') \
                     .view(B, -1).sum(dim=1)
        else:
            log_p = -F.mse_loss(x_recon, y, reduction='none') \
                     .view(B, -1).sum(dim=1)
        recon_loglik.append(log_p)
    return torch.stack(recon_loglik, dim=1), torch.stack(recon_images, dim=1)

def compute_NMI(Z, Y, n_clusters=10):
    try:
        if len(Z) < n_clusters:
            return 0.0
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        return float(NMI(Y, km.fit_predict(Z)))
    except:
        return 0.0

def compute_posterior_accuracy(preds, true_labels, K, num_true=10):
    """
    K == num_true: 标准 Hungarian 匹配 (一一对应)
    K >  num_true: 过聚类 (over-clustering), 用多数投票法,
                   多个 cluster 可以对应同一个真实类别
    """
    if K <= num_true:
        # 原始 Hungarian 方法
        cost = np.zeros((num_true, K))
        for i in range(num_true):
            for j in range(K):
                cost[i, j] = -np.sum((true_labels == i) & (preds == j))
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    else:
        # 多数投票: 每个 cluster 对应其内部最多的真实标签
        mapping = {}
        for k in range(K):
            mask = preds == k
            if mask.sum() > 0:
                mapping[k] = int(np.bincount(true_labels[mask],
                                              minlength=num_true).argmax())
            else:
                mapping[k] = 0

    aligned = np.array([mapping.get(p, 0) for p in preds])
    return np.mean(aligned == true_labels), mapping


class TrainingLogger:
    KEYS = ["epoch", "phase", "loss", "recon_loss", "kl_loss",
            "prior_loss", "posterior_corr", "beta", "tau",
            "nmi", "posterior_acc", "resp_entropy",
            "pi_entropy", "pi_values", "balance_loss"]

    def __init__(self):
        self.records = {k: [] for k in self.KEYS}

    def log(self, **kwargs):
        for k in self.KEYS:
            self.records[k].append(kwargs.get(k, None))

    def save(self, path):
        out = {}
        for k, v in self.records.items():
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)