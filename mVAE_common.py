# mVAE_common.py
# 公共组件：网络结构、工具函数、可视化（严格对齐论文 Section 2.2）

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
    latent_dim = 2                 # ★★ 4→2: z 只有2维,必须依赖 x
    hidden_dim = 256
    num_classes = 10

    # --- 论文公式参数 ---
    beta = 2.0                     # ★★ 0.5→2.0: 强力压缩 z
    use_bce = True

    # --- Gumbel softmax ---
    init_gumbel_temp = 0.3         # ★★ 0.5→0.3: 从一开始就接近 one-hot
    min_gumbel_temp = 0.05
    gumbel_anneal_rate = 0.97      # ★★ 0.98→0.97: 更快退火
    current_gumbel_temp = init_gumbel_temp

    # --- 半监督 ---
    alpha_unlabeled = 0.5
    labeled_per_class = 100

    # --- 训练 ---
    lr = 1e-3
    batch_size = 128
    optuna_epochs = 15
    final_epochs = 80
    kl_anneal_epochs = 5           # ★★ 20→5: β 在 5 个 epoch 内就到位

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
# Conditional Decoder  p(y|x,z,θ)
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
# 工具函数
# ============================================================
def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

def gumbel_softmax_sample(logits, temperature):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return F.softmax((logits + gumbel) / (temperature + 1e-9), dim=-1)

def compute_recon_loglik(dec, z, y, K, use_bce=True):
    """对每个类 k 计算 log p(y|x=k, z, θ)，返回 [B, K]"""
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
            # BCE: log Bernoulli(y; f_θ(k,z))
            log_p = -F.binary_cross_entropy(x_recon, y, reduction='none') \
                     .view(B, -1).sum(dim=1)
        else:
            # MSE (Gaussian): -0.5 * ||y - f_θ(k,z)||²
            log_p = -F.mse_loss(x_recon, y, reduction='none') \
                     .view(B, -1).sum(dim=1)
        recon_loglik.append(log_p)
    recon_loglik = torch.stack(recon_loglik, dim=1)   # [B, K]
    recon_images = torch.stack(recon_images, dim=1)    # [B, K, C, H, W]
    return recon_loglik, recon_images


# ============================================================
# NMI via KMeans
# ============================================================
def compute_NMI(Z, Y, n_clusters=10):
    try:
        if len(Z) < n_clusters:
            return 0.0
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        return float(NMI(Y, km.fit_predict(Z)))
    except:
        return 0.0


# ============================================================
# Posterior accuracy (匈牙利对齐)
# ============================================================
def compute_posterior_accuracy(preds, true_labels, K):
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = -np.sum((true_labels == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned = np.array([mapping.get(p, 0) for p in preds])
    return np.mean(aligned == true_labels), mapping


# ============================================================
# TrainingLogger — 记录训练全过程指标
# ============================================================
class TrainingLogger:
    KEYS = ["epoch", "phase", "loss", "recon_loss", "kl_loss",
            "prior_loss", "posterior_corr", "beta", "tau",
            "nmi", "posterior_acc", "resp_entropy",
            "pi_entropy", "pi_values"]

    def __init__(self):
        self.records = {k: [] for k in self.KEYS}

    def log(self, **kwargs):
        for k in self.KEYS:
            self.records[k].append(kwargs.get(k, None))

    def save(self, path):
        # numpy arrays -> list for JSON
        out = {}
        for k, v in self.records.items():
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    def load(self, path):
        with open(path) as f:
            self.records = json.load(f)
