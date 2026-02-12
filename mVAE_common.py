# mVAE_common.py
# 公共组件：网络结构、工具函数、可视化（严格对齐论文 Section 2.2）
# ★★ v2 修改: 修复 x-collapse 问题

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
    latent_dim = 2                 # ★★ v2: 4→2, 更强瓶颈迫使 x 被使用
    hidden_dim = 256
    num_classes = 10

    # --- 论文公式参数 ---
    beta = 1.0                     # 最终 β 目标值
    beta_init = 5.0                # ★★ v2: 3→5, 更强逆向退火
    use_bce = True

    # --- Gumbel softmax ---
    init_gumbel_temp = 0.5         # ★★ v2: 0.3→0.5, 初期稍软以便学习
    min_gumbel_temp = 0.05
    gumbel_anneal_rate = 0.97
    current_gumbel_temp = init_gumbel_temp

    # --- ★★ v2 新增: 打破 x-collapse 的参数 ---
    hard_gumbel_epochs = 20        # 前 N 个 epoch 用 straight-through hard Gumbel
    logit_mix_alpha = 0.1          # 混合 non-detached logits 的比例 (0=纯论文, 1=全 live)

    # --- 半监督 ---
    alpha_unlabeled = 0.5
    labeled_per_class = 100

    # --- 训练 ---
    lr = 1e-3
    batch_size = 128
    optuna_epochs = 15
    final_epochs = 80
    kl_anneal_epochs = 20          # ★★ v2: 15→20, 配合更强 beta_init
    decoder_type = 'film'          # 'concat'=旧版, 'film'=FiLM条件化

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
# Conditional Decoder (Concatenation — 原版, 弱条件化)
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
# FiLM Decoder (★★ 强条件化 — decoder 结构性依赖 x)
# ============================================================
class FiLMDecoder(nn.Module):
    def __init__(self, latent_dim=4, num_classes=10, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 类嵌入: x → 中间表示
        self.class_embed = nn.Sequential(
            nn.Linear(num_classes, 64), nn.ReLU(), nn.Linear(64, 64))

        # z 到初始特征
        self.z_fc = nn.Linear(latent_dim, hidden_dim)

        # FiLM 层 1: class → scale+shift for hidden_dim
        self.film1_gamma = nn.Linear(64, hidden_dim)
        self.film1_beta = nn.Linear(64, hidden_dim)

        # hidden → spatial
        self.fc2 = nn.Linear(hidden_dim, 64 * 7 * 7)

        # FiLM 层 2: class → scale+shift for conv channels
        self.film2_gamma = nn.Linear(64, 64)
        self.film2_beta = nn.Linear(64, 64)

        # 反卷积
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        )

        # FiLM 层 3: class → scale+shift for conv channels
        self.film3_gamma = nn.Linear(64, 32)
        self.film3_beta = nn.Linear(64, 32)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z, y_onehot):
        # 类嵌入
        c = self.class_embed(y_onehot)  # [B, 64]

        # z → 初始特征
        h = F.relu(self.z_fc(z))  # [B, hidden_dim]

        # FiLM 调制 1: γ(x) ⊙ h + β(x)
        gamma1 = self.film1_gamma(c)  # [B, hidden_dim]
        beta1 = self.film1_beta(c)
        h = gamma1 * h + beta1
        h = F.relu(h)

        # → spatial
        h = F.relu(self.fc2(h))  # [B, 64*7*7]
        h = h.view(-1, 64, 7, 7)

        # FiLM 调制 2
        gamma2 = self.film2_gamma(c).unsqueeze(-1).unsqueeze(-1)  # [B,64,1,1]
        beta2 = self.film2_beta(c).unsqueeze(-1).unsqueeze(-1)
        h = gamma2 * h + beta2
        h = F.relu(h)

        # 反卷积
        h = self.deconv(h)  # [B, 32, 14, 14]

        # FiLM 调制 3
        gamma3 = self.film3_gamma(c).unsqueeze(-1).unsqueeze(-1)  # [B,32,1,1]
        beta3 = self.film3_beta(c).unsqueeze(-1).unsqueeze(-1)
        h = gamma3 * h + beta3
        h = F.relu(h)

        return self.final_conv(h)  # [B, 1, 28, 28]


# ============================================================
# 工具函数
# ============================================================
def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def gumbel_softmax_sample(logits, temperature, hard=False):
    """
    Gumbel softmax 采样。

    ★★ v2 新增 hard 模式:
    - hard=False: 标准 soft Gumbel (原版)
    - hard=True:  Straight-through Gumbel
                  前向: one-hot (明确分配到某一类)
                  反向: soft 梯度 (可微)
    """
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    y_soft = F.softmax((logits + gumbel) / (temperature + 1e-9), dim=-1)

    if hard:
        # Straight-through estimator
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft  # 前向=one-hot, 反向=soft梯度
    return y_soft


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
            log_p = -F.binary_cross_entropy(x_recon, y, reduction='none') \
                     .view(B, -1).sum(dim=1)
        else:
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
        out = {}
        for k, v in self.records.items():
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    def load(self, path):
        with open(path) as f:
            self.records = json.load(f)
