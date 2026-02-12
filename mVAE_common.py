# mVAE_common.py
# 公共组件：网络结构、工具函数
# ★★ v3: 借鉴 HMM-VAE 的成功策略

import os, json, time
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
    # --- 模型 ---
    latent_dim = 2
    hidden_dim = 256
    num_classes = 10

    # --- 论文公式参数 ---
    beta = 2.0                     # Phase 2 最终 β
    use_bce = True

    # --- Gumbel softmax ---
    init_gumbel_temp = 2.0         # ★★ v3: 跟 HMM-VAE 一致
    min_gumbel_temp = 0.1
    gumbel_anneal_rate = 0.98
    current_gumbel_temp = init_gumbel_temp

    # --- ★★ v3: 两阶段训练 (借鉴 HMM-VAE) ---
    pretrain_epochs = 30           # Phase 1: emission bootstrap
    pretrain_beta = 50.0           # Phase 1 的 KL 权重 (压制 z)
    z_dropout_rate = 0.7           # Phase 1 的 z dropout
    balance_weight = 50.0          # 防止类坍缩
    pretrain_lr = 1e-3             # Phase 1 学习率
    finetune_lr = 1e-4             # Phase 2 学习率 (VAE 部分)
    pi_lr = 1e-2                   # Phase 2 Π 学习率

    # --- 半监督 ---
    alpha_unlabeled = 0.5
    labeled_per_class = 100

    # --- 训练 ---
    lr = 1e-3
    batch_size = 128
    optuna_epochs = 15
    final_epochs = 100
    decoder_type = 'film'

    # --- 设备 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "./mVAE_results"


# ============================================================
# Encoder  q_ϕ(z|y) = N(z; μ_ϕ(y), σ²_ϕ(y)I)
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
# FiLM Decoder (强条件化)
# ============================================================
class FiLMDecoder(nn.Module):
    def __init__(self, latent_dim=4, num_classes=10, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.class_embed = nn.Sequential(
            nn.Linear(num_classes, 64), nn.ReLU(), nn.Linear(64, 64))
        self.z_fc = nn.Linear(latent_dim, hidden_dim)

        self.film1_gamma = nn.Linear(64, hidden_dim)
        self.film1_beta = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64 * 7 * 7)
        self.film2_gamma = nn.Linear(64, 64)
        self.film2_beta = nn.Linear(64, 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU())
        self.film3_gamma = nn.Linear(64, 32)
        self.film3_beta = nn.Linear(64, 32)
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, z, y_onehot):
        c = self.class_embed(y_onehot)
        h = F.relu(self.z_fc(z))

        gamma1 = self.film1_gamma(c)
        beta1 = self.film1_beta(c)
        h = gamma1 * h + beta1
        h = F.relu(h)

        h = F.relu(self.fc2(h))
        h = h.view(-1, 64, 7, 7)

        gamma2 = self.film2_gamma(c).unsqueeze(-1).unsqueeze(-1)
        beta2 = self.film2_beta(c).unsqueeze(-1).unsqueeze(-1)
        h = gamma2 * h + beta2
        h = F.relu(h)

        h = self.deconv(h)

        gamma3 = self.film3_gamma(c).unsqueeze(-1).unsqueeze(-1)
        beta3 = self.film3_beta(c).unsqueeze(-1).unsqueeze(-1)
        h = gamma3 * h + beta3
        h = F.relu(h)

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
    recon_loglik = torch.stack(recon_loglik, dim=1)
    recon_images = torch.stack(recon_images, dim=1)
    return recon_loglik, recon_images


def compute_NMI(Z, Y, n_clusters=10):
    try:
        if len(Z) < n_clusters:
            return 0.0
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        return float(NMI(Y, km.fit_predict(Z)))
    except:
        return 0.0


def compute_posterior_accuracy(preds, true_labels, K):
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = -np.sum((true_labels == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
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

    def load(self, path):
        with open(path) as f:
            self.records = json.load(f)
