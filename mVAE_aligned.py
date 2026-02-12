# mVAE_aligned.py
# ═══════════════════════════════════════════════════════════════
# Mixture VAE — 严格对齐论文 Section 2.2 的 Partially Variational EM
#
# 支持两种模式:  --mode unsup     (纯无监督)
#               --mode semisup   (半监督)
#
# 用法:
#   python mVAE_aligned.py --mode unsup   --epochs 60
#   python mVAE_aligned.py --mode semisup --epochs 60 --labeled_per_class 100
#   python mVAE_aligned.py --optuna        (先调参再训练)
#
# 论文公式 (Section 2.2.1):
#   -J^(t) = -log p(y|x,z,θ) - log p(x|Π) + KL(q(z|y)||p(z))
#            + log p(x|z,y,θ^(t),Π^(t))
#
# 修正要点 (相对于原代码):
#   1. 重建 loss: 对 log p(y|x=k,z) 加权求和 (而非对图像混合后算 MSE)
#   2. 加入 prior 项: -Σ_k resp_k · log π_k
#   3. 加入 posterior correction 项: +Σ_k resp_k · log p(x=k|z,y,θ^(t))
#   4. Π 作为可学习参数 (nn.Parameter) 通过梯度更新
#   5. 后验计算使用 detached 参数 (近似 θ^(t))
#   6. 使用 BCE (Bernoulli) 更接近论文的 Cat
#   7. 不需要额外的 entropy 正则 (③④合并后自然防坍缩)
# ═══════════════════════════════════════════════════════════════

import os, sys, json, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from mVAE_common import (
    Config, Encoder, ConditionalDecoder,
    reparameterize, gumbel_softmax_sample, compute_recon_loglik,
    compute_NMI, compute_posterior_accuracy, TrainingLogger
)

# ============================================================
# Model: mVAE (Paper-Aligned)
# ============================================================
class mVAE(nn.Module):
    """
    Mixture VAE with partially variational EM.

    论文符号:
      y = 观测 (图像)
      x ∈ {1,...,K} = 离散类别 (推断目标)
      z = 连续潜变量
      Π = (π_1,...,π_K) = 类先验 (可学习)
      θ = decoder 参数
      ϕ = encoder 参数
    """
    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(cfg.latent_dim)
        self.dec = ConditionalDecoder(cfg.latent_dim, cfg.num_classes, cfg.hidden_dim)
        # Π 作为可学习参数 (通过 softmax 归一化)
        self.log_pi = nn.Parameter(torch.zeros(cfg.num_classes))
        self.K = cfg.num_classes

    @property
    def pi(self):
        return F.softmax(self.log_pi, dim=0)

    def forward_labeled(self, y, x_true, cfg):
        """
        有标签数据: x 已知, 标准 conditional VAE loss

        loss = -E_q[log p(y|x,z,θ)] + β·KL(q(z|y)||p(z)) - log π_x
        """
        mu, logvar = self.enc(y)
        z = reparameterize(mu, logvar)
        B = y.size(0)

        # 重建
        x_onehot = F.one_hot(x_true, self.K).float()
        y_recon = self.dec(z, x_onehot)

        if cfg.use_bce:
            recon_loss = F.binary_cross_entropy(y_recon, y, reduction='sum') / B
        else:
            recon_loss = F.mse_loss(y_recon, y, reduction='sum') / B

        # KL
        kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Prior: -log π_{x_true}
        log_pi = torch.log(self.pi + 1e-9)
        prior_loss = -log_pi[x_true].mean()

        loss = recon_loss + cfg.beta * kl_z + prior_loss
        return loss, {
            'recon': recon_loss.item(), 'kl': kl_z.item(),
            'prior': prior_loss.item(), 'post_corr': 0.0,
            'resp_ent': 0.0, 'mu': mu.detach()
        }

    def forward_unlabeled(self, y, cfg):
        """
        无标签数据: x 需要推断, 使用 partially variational EM

        论文公式:
        -J^(t) = -Σ_k x_k log p(y|k,z,θ)     [① 加权重建]
                 -Σ_k x_k log π_k              [② 先验]
                 + KL(q(z|y)||p(z))             [③ KL散度]
                 +Σ_k x_k log p(x=k|z,y,θ^(t)) [④ 后验校正]

        其中 x 通过 Gumbel softmax 从后验 p(x|z,y,θ^(t)) 采样
        """
        mu, logvar = self.enc(y)
        z = reparameterize(mu, logvar)
        B = y.size(0)

        # === 计算每个类的 log p(y|x=k, z, θ) ===
        # 这里需要梯度 (用于 M-step 更新 θ)
        recon_loglik, _ = compute_recon_loglik(
            self.dec, z, y, self.K, cfg.use_bce)  # [B, K]

        # === E-step: 计算后验 p(x|z,y,θ^(t),Π^(t)) ===
        # 使用 detached 值, 近似 θ^(t)
        log_pi = torch.log(self.pi + 1e-9)                       # [K]
        logits = log_pi.unsqueeze(0) + recon_loglik.detach()      # [B, K], detached!
        log_posterior = F.log_softmax(logits, dim=1)              # log p(x=k|z,y,θ^(t))

        # Gumbel softmax 采样 x
        resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp)  # [B, K]

        # === 计算论文的 -J^(t) 各项 ===

        # ① 加权重建: -Σ_k x_k · log p(y|k,z,θ)
        weighted_recon = -(resp * recon_loglik).sum(dim=1).mean()

        # ② 先验: -Σ_k x_k · log π_k
        prior_loss = -(resp * log_pi.unsqueeze(0)).sum(dim=1).mean()

        # ③ KL 散度: KL(q_ϕ(z|y) || p(z))
        kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # ④ 后验校正: +Σ_k x_k · log p(x=k|z,y,θ^(t))
        # 注意: log_posterior 是 detached 的, 梯度只通过 resp 传
        posterior_corr = (resp * log_posterior).sum(dim=1).mean()

        # 总 loss = -J^(t) 的 Monte Carlo 估计
        loss = weighted_recon + cfg.beta * kl_z + prior_loss + posterior_corr

        # 诊断指标
        resp_entropy = -(resp * torch.log(resp + 1e-9)).sum(dim=1).mean()

        return loss, {
            'recon': weighted_recon.item(), 'kl': kl_z.item(),
            'prior': prior_loss.item(), 'post_corr': posterior_corr.item(),
            'resp_ent': resp_entropy.item(), 'mu': mu.detach(),
            'resp': resp.detach()
        }


# ============================================================
# Data Loaders
# ============================================================
def get_unsup_loaders(cfg):
    ds = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())
    n_train = int(0.9 * len(ds))
    train_set, val_set = random_split(ds, [n_train, len(ds) - n_train],
                                       generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


def get_semisup_loaders(cfg):
    ds = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())
    labels = np.array(ds.targets)
    labeled_idx, unlabeled_idx = [], []
    for c in range(cfg.num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.seed(42)
        np.random.shuffle(idx_c)
        labeled_idx.extend(idx_c[:cfg.labeled_per_class])
        unlabeled_idx.extend(idx_c[cfg.labeled_per_class:])

    labeled_set = Subset(ds, labeled_idx)
    unlabeled_set = Subset(ds, unlabeled_idx)
    val_set = Subset(ds, list(range(int(0.1 * len(ds)))))

    return (
        DataLoader(labeled_set, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(unlabeled_set, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False),
    )


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, cfg):
    """返回 NMI, posterior_acc, avg_loss"""
    model.eval()
    zs, ys_true, preds = [], [], []
    total_loss = 0
    n_batches = 0

    for y_img, y_label in loader:
        y_img = y_img.to(cfg.device)
        loss, info = model.forward_unlabeled(y_img, cfg)
        total_loss += loss.item()
        n_batches += 1

        zs.append(info['mu'].cpu().numpy())
        ys_true.append(y_label.numpy())

        if 'resp' in info and info['resp'] is not None:
            preds.append(info['resp'].argmax(dim=1).cpu().numpy())

    zs = np.concatenate(zs)
    ys_true = np.concatenate(ys_true)
    nmi = compute_NMI(zs, ys_true, cfg.num_classes)

    post_acc = 0.0
    if preds:
        preds = np.concatenate(preds)
        post_acc, _ = compute_posterior_accuracy(preds, ys_true, cfg.num_classes)

    return nmi, post_acc, total_loss / max(n_batches, 1)


# ============================================================
# Diagnostic: x-conditionality score
# ============================================================
@torch.no_grad()
def measure_x_conditionality(model, cfg, n_z=20):
    """
    测量 decoder 对 x 的依赖程度。
    固定 z, 改变 x, 看输出变化有多大。
    返回 0~1 之间的分数: 0=完全忽略x, 1=强依赖x

    这是最关键的诊断指标 — 如果 xcond≈0 说明 x 坍缩了。
    """
    model.eval()
    K = cfg.num_classes
    z = torch.randn(n_z, cfg.latent_dim).to(cfg.device)

    outputs = []
    for k in range(K):
        y_onehot = F.one_hot(
            torch.full((n_z,), k, device=cfg.device, dtype=torch.long), K).float()
        out = model.dec(z, y_onehot)
        outputs.append(out)
    outputs = torch.stack(outputs, dim=0)  # [K, n_z, C, H, W]

    # 对每个 z 样本, 计算不同 x 输出之间的标准差
    # outputs[:, i, :] 是同一个 z_i 在不同 x 下的 K 个输出
    std_across_x = outputs.std(dim=0)  # [n_z, C, H, W]
    mean_std = std_across_x.mean().item()

    # 归一化: 用整体像素值的标准差做参考
    pixel_std = outputs.std().item() + 1e-9
    score = mean_std / pixel_std

    return min(score, 1.0)


# ============================================================
# Training: Unsupervised
# ============================================================
def train_unsupervised(model, optimizer, train_loader, val_loader, cfg,
                       logger=None, is_final=False):
    total_epochs = cfg.final_epochs if is_final else cfg.optuna_epochs
    beta_target = cfg.beta
    best_nmi = 0.0
    best_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        ep_recon, ep_kl, ep_prior, ep_post, ep_ent = 0, 0, 0, 0, 0
        n_batches = 0

        # Beta annealing (线性 warmup)
        if is_final and epoch <= cfg.kl_anneal_epochs:
            cfg.beta = beta_target * (epoch / cfg.kl_anneal_epochs)
        else:
            cfg.beta = beta_target

        # ★ Tau annealing: 从 epoch 1 就开始退火! (不再延迟)
        cfg.current_gumbel_temp = max(
            cfg.min_gumbel_temp,
            cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        for y_img, _ in train_loader:
            y_img = y_img.to(cfg.device)
            loss, info = model.forward_unlabeled(y_img, cfg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            ep_recon += info['recon']
            ep_kl += info['kl']
            ep_prior += info['prior']
            ep_post += info['post_corr']
            ep_ent += info['resp_ent']
            n_batches += 1

        # Evaluate
        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)

        # ★ 用 post_acc 选 best model (而非 NMI — NMI 高不代表 x 被使用)
        if post_acc > best_acc:
            best_acc = post_acc
            if is_final:
                torch.save(model.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
        best_nmi = max(best_nmi, nmi)

        pi_np = model.pi.detach().cpu().numpy()

        # ★ 诊断: x-conditionality score (decoder 对不同 x 的差异有多大)
        x_cond = 0.0
        if is_final and epoch % 5 == 0:
            x_cond = measure_x_conditionality(model, cfg)

        if logger:
            logger.log(
                epoch=epoch, phase="Unsup-Final" if is_final else "Unsup-Optuna",
                loss=val_loss,
                recon_loss=ep_recon / n_batches,
                kl_loss=ep_kl / n_batches,
                prior_loss=ep_prior / n_batches,
                posterior_corr=ep_post / n_batches,
                beta=cfg.beta, tau=cfg.current_gumbel_temp,
                nmi=nmi, posterior_acc=post_acc,
                resp_entropy=ep_ent / n_batches,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                pi_values=pi_np,
            )

        if is_final:
            cond_str = f" xcond={x_cond:.3f}" if x_cond > 0 else ""
            print(f"  Ep {epoch:3d}/{total_epochs} | NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep_recon/n_batches:.1f} KL={ep_kl/n_batches:.2f} "
                  f"β={cfg.beta:.2f} τ={cfg.current_gumbel_temp:.3f}{cond_str}")

    cfg.beta = beta_target
    return best_nmi


# ============================================================
# Training: Semi-supervised
# ============================================================
def train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
                         val_loader, cfg, logger=None, is_final=False):
    total_epochs = cfg.final_epochs if is_final else cfg.optuna_epochs
    beta_target = cfg.beta
    best_nmi = 0.0
    best_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        ep_recon, ep_kl, ep_prior, ep_post, ep_ent = 0, 0, 0, 0, 0
        n_batches = 0

        # Beta annealing (线性 warmup)
        if is_final and epoch <= cfg.kl_anneal_epochs:
            cfg.beta = beta_target * (epoch / cfg.kl_anneal_epochs)
        else:
            cfg.beta = beta_target

        # ★ Tau annealing: 从 epoch 1 就开始! (不再延迟)
        cfg.current_gumbel_temp = max(
            cfg.min_gumbel_temp,
            cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
            x_lab = x_lab.to(cfg.device)
            y_lab = y_lab.to(cfg.device)
            x_un = x_un.to(cfg.device)

            loss_lab, info_lab = model.forward_labeled(x_lab, y_lab, cfg)
            loss_un, info_un = model.forward_unlabeled(x_un, cfg)

            loss = loss_lab + cfg.alpha_unlabeled * loss_un

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            ep_recon += (info_lab['recon'] + info_un['recon']) / 2
            ep_kl += (info_lab['kl'] + info_un['kl']) / 2
            ep_prior += (info_lab['prior'] + info_un['prior']) / 2
            ep_post += info_un['post_corr']
            ep_ent += info_un['resp_ent']
            n_batches += 1

        # Evaluate
        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)

        # ★ 用 post_acc 选 best model
        if post_acc > best_acc:
            best_acc = post_acc
            if is_final:
                torch.save(model.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
        best_nmi = max(best_nmi, nmi)

        pi_np = model.pi.detach().cpu().numpy()

        # ★ 诊断: x-conditionality
        x_cond = 0.0
        if is_final and epoch % 5 == 0:
            x_cond = measure_x_conditionality(model, cfg)

        if logger:
            logger.log(
                epoch=epoch, phase="SemiSup-Final" if is_final else "SemiSup-Optuna",
                loss=val_loss,
                recon_loss=ep_recon / n_batches,
                kl_loss=ep_kl / n_batches,
                prior_loss=ep_prior / n_batches,
                posterior_corr=ep_post / n_batches,
                beta=cfg.beta, tau=cfg.current_gumbel_temp,
                nmi=nmi, posterior_acc=post_acc,
                resp_entropy=ep_ent / n_batches,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                pi_values=pi_np,
            )

        if is_final:
            cond_str = f" xcond={x_cond:.3f}" if x_cond > 0 else ""
            print(f"  Ep {epoch:3d}/{total_epochs} | NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep_recon/n_batches:.1f} KL={ep_kl/n_batches:.2f} "
                  f"β={cfg.beta:.2f} τ={cfg.current_gumbel_temp:.3f}{cond_str}")

    cfg.beta = beta_target
    return best_nmi


# ============================================================
# Visualization System (16 张图)
# ============================================================
# ---- 全局样式 ----
COLORS = ['#2c73d2', '#ff6b6b', '#51cf66', '#ffa94d', '#845ef7',
          '#f06595', '#20c997', '#fab005', '#339af0', '#ff8787']

def _setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 12,
        'figure.dpi': 150, 'savefig.dpi': 200,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })
_setup_style()


def fig01_training_curves(logger, save_path, mode="unsup"):
    """4-panel: Loss / Recon+KL / NMI+Acc / τ+β"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs = logger.records['epoch']
    n = len(epochs)

    # Panel 1: Total loss
    ax = axes[0, 0]
    ax.plot(epochs, logger.records['loss'], color=COLORS[0], linewidth=1.5)
    ax.set_title("Validation Loss"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    # Panel 2: Recon + KL + Prior + PostCorr
    ax = axes[0, 1]
    ax.plot(epochs, logger.records['recon_loss'], label='Recon', color=COLORS[0])
    ax.plot(epochs, logger.records['kl_loss'], label='KL', color=COLORS[1])
    if any(v and v != 0 for v in logger.records['prior_loss']):
        ax.plot(epochs, logger.records['prior_loss'], label='Prior', color=COLORS[2])
    if any(v and v != 0 for v in logger.records['posterior_corr']):
        ax.plot(epochs, logger.records['posterior_corr'], label='PostCorr', color=COLORS[3])
    ax.legend(fontsize=9); ax.set_title("Loss Components"); ax.grid(alpha=0.3)

    # Panel 3: NMI + Acc
    ax = axes[1, 0]
    ax.plot(epochs, logger.records['nmi'], label='NMI', color=COLORS[0], linewidth=2)
    ax.plot(epochs, logger.records['posterior_acc'], label='Post.Acc', color=COLORS[1],
            linewidth=2, linestyle='--')
    ax.legend(fontsize=10); ax.set_title("Clustering Quality")
    ax.set_xlabel("Epoch"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    # Panel 4: τ + β
    ax = axes[1, 1]
    ax.plot(epochs, logger.records['tau'], label='τ (Gumbel)', color=COLORS[4])
    ax2 = ax.twinx()
    ax2.plot(epochs, logger.records['beta'], label='β (KL)', color=COLORS[3], linestyle='--')
    ax.set_xlabel("Epoch"); ax.set_ylabel("τ"); ax2.set_ylabel("β")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.set_title("Schedule: τ & β"); ax.grid(alpha=0.3)

    fig.suptitle(f"mVAE Training ({mode})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ fig01_training_curves → {save_path}")


def fig02_nmi_acc_detail(logger, save_path):
    """NMI + Acc 大图, 标注最佳值"""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = logger.records['epoch']
    nmis = logger.records['nmi']
    accs = logger.records['posterior_acc']

    ax.plot(epochs, nmis, label='NMI', color=COLORS[0], linewidth=2)
    ax.plot(epochs, accs, label='Posterior Acc', color=COLORS[1], linewidth=2)

    best_nmi_idx = np.argmax(nmis)
    best_acc_idx = np.argmax(accs)
    ax.annotate(f"Best NMI={nmis[best_nmi_idx]:.4f}",
                xy=(epochs[best_nmi_idx], nmis[best_nmi_idx]),
                fontsize=9, color=COLORS[0],
                arrowprops=dict(arrowstyle='->', color=COLORS[0]),
                xytext=(epochs[best_nmi_idx]+2, nmis[best_nmi_idx]-0.08))
    ax.annotate(f"Best Acc={accs[best_acc_idx]:.4f}",
                xy=(epochs[best_acc_idx], accs[best_acc_idx]),
                fontsize=9, color=COLORS[1],
                arrowprops=dict(arrowstyle='->', color=COLORS[1]),
                xytext=(epochs[best_acc_idx]+2, accs[best_acc_idx]+0.05))

    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=12); ax.grid(alpha=0.3)
    ax.set_title("Clustering Quality Over Training", fontsize=14)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig02_nmi_acc_detail → {save_path}")


def fig03_resp_entropy(logger, save_path):
    """resp entropy + π entropy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = logger.records['epoch']

    ax1.plot(epochs, logger.records['resp_entropy'], color=COLORS[4])
    ax1.axhline(y=np.log(10), color='gray', linestyle=':', label='Max entropy (ln10)')
    ax1.set_title("Responsibility Entropy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, logger.records['pi_entropy'], color=COLORS[2])
    ax2.axhline(y=np.log(10), color='gray', linestyle=':', label='Uniform (ln10)')
    ax2.set_title("π Entropy (Prior)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig03_resp_entropy → {save_path}")


def fig04_pi_evolution(logger, save_path, K=10):
    """π 随训练变化 (stacked area)"""
    pi_list = [v for v in logger.records['pi_values'] if v is not None]
    if not pi_list:
        return
    pi_arr = np.array(pi_list)  # [n_epochs, K]
    epochs = list(range(1, len(pi_arr) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(epochs, pi_arr.T, labels=[f"Class {k}" for k in range(K)],
                 colors=COLORS[:K], alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("π_k")
    ax.set_title("Prior Π Evolution Over Training")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_xlim(epochs[0], epochs[-1])
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig04_pi_evolution → {save_path}")


def fig05_latent_space(model, loader, cfg, save_path, method='pca'):
    """潜在空间散点图"""
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for y_img, y_label in loader:
            mu, _ = model.enc(y_img.to(cfg.device))
            zs.append(mu.cpu().numpy())
            ys.append(y_label.numpy())
    zs = np.concatenate(zs)
    ys = np.concatenate(ys)

    fig, ax = plt.subplots(figsize=(8, 7))
    if zs.shape[1] == 2:
        coords = zs
        xlabel, ylabel = "z₁", "z₂"
    else:
        if method == 'tsne':
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(zs[:5000])
            ys_plot = ys[:5000]
            xlabel, ylabel = "t-SNE 1", "t-SNE 2"
        else:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(zs)
            ys_plot = ys
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"

    if zs.shape[1] != 2:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=ys_plot, cmap='tab10',
                            s=5, alpha=0.5)
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=ys, cmap='tab10',
                            s=5, alpha=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"Latent Space ({method.upper()})")
    plt.colorbar(scatter, ax=ax, label="True class")
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig05_latent_space → {save_path}")


def fig06_generated_samples(model, cfg, save_path, n_per_class=10):
    """每类生成样本"""
    model.eval()
    z = torch.randn(n_per_class, cfg.latent_dim).to(cfg.device)
    all_samples = []
    with torch.no_grad():
        for k in range(cfg.num_classes):
            y_onehot = F.one_hot(
                torch.full((n_per_class,), k, device=cfg.device, dtype=torch.long),
                cfg.num_classes).float()
            samples = model.dec(z, y_onehot)
            all_samples.append(samples)
    grid = torch.cat(all_samples, dim=0)
    save_image(grid, save_path, nrow=n_per_class, normalize=True)
    print(f"  ✓ fig06_generated_samples → {save_path}")


def fig07_confusion_matrix(model, loader, cfg, save_path):
    """聚类混淆矩阵"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for y_img, y_label in loader:
            y_img = y_img.to(cfg.device)
            _, info = model.forward_unlabeled(y_img, cfg)
            if 'resp' in info and info['resp'] is not None:
                preds.append(info['resp'].argmax(dim=1).cpu().numpy())
                trues.append(y_label.numpy())
    if not preds:
        return
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    K = cfg.num_classes
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Raw counts
    im1 = ax1.imshow(cm, cmap='Blues')
    ax1.set_xlabel("Predicted cluster"); ax1.set_ylabel("True class")
    ax1.set_title("Confusion Matrix (Counts)")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Row-normalized
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im2 = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xlabel("Predicted cluster"); ax2.set_ylabel("True class")
    ax2.set_title("Confusion Matrix (Row-normalized)")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    for ax in [ax1, ax2]:
        ax.set_xticks(range(K)); ax.set_yticks(range(K))

    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig07_confusion_matrix → {save_path}")


def fig08_cluster_distribution(model, loader, cfg, save_path):
    """聚类分布 vs 真实分布"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for y_img, y_label in loader:
            y_img = y_img.to(cfg.device)
            _, info = model.forward_unlabeled(y_img, cfg)
            if 'resp' in info and info['resp'] is not None:
                preds.append(info['resp'].argmax(dim=1).cpu().numpy())
                trues.append(y_label.numpy())
    if not preds:
        return
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    K = cfg.num_classes

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(K)
    width = 0.35
    true_freq = np.bincount(trues, minlength=K).astype(float)
    true_freq /= true_freq.sum()
    pred_freq = np.bincount(preds, minlength=K).astype(float)
    pred_freq /= pred_freq.sum()

    ax.bar(x_pos - width/2, true_freq, width, label='True', color=COLORS[0], alpha=0.7)
    ax.bar(x_pos + width/2, pred_freq, width, label='Predicted', color=COLORS[1], alpha=0.7)
    ax.set_xlabel("Class/Cluster"); ax.set_ylabel("Frequency")
    ax.set_xticks(x_pos); ax.legend()
    ax.set_title("Class Distribution: True vs Predicted")
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig08_cluster_distribution → {save_path}")


def fig09_reconstruction(model, loader, cfg, save_path, n_show=8):
    """原图 vs 重建 对比"""
    model.eval()
    with torch.no_grad():
        y_img, y_label = next(iter(loader))
        y_img = y_img[:n_show].to(cfg.device)
        y_label = y_label[:n_show]

        mu, logvar = model.enc(y_img)
        z = mu  # 用 mu 而非采样

        # 用真实标签重建
        y_onehot = F.one_hot(y_label.to(cfg.device), cfg.num_classes).float()
        y_recon = model.dec(z, y_onehot)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.5, 3.5))
    for i in range(n_show):
        axes[0, i].imshow(y_img[i, 0].cpu(), cmap='gray'); axes[0, i].axis('off')
        axes[1, i].imshow(y_recon[i, 0].cpu(), cmap='gray'); axes[1, i].axis('off')
        axes[0, i].set_title(f"y={y_label[i].item()}", fontsize=9)
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)
    fig.suptitle("Reconstruction Quality", fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig09_reconstruction → {save_path}")


def fig10_loss_decomposition(logger, save_path):
    """Stacked area: 各 loss 成分比例"""
    epochs = logger.records['epoch']
    recon = np.abs(logger.records['recon_loss'])
    kl = np.abs(logger.records['kl_loss'])
    prior = np.abs([v if v else 0 for v in logger.records['prior_loss']])
    post = np.abs([v if v else 0 for v in logger.records['posterior_corr']])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(epochs, recon, kl, prior, post,
                 labels=['|Recon|', '|KL|', '|Prior|', '|PostCorr|'],
                 colors=[COLORS[0], COLORS[1], COLORS[2], COLORS[3]], alpha=0.7)
    ax.legend(loc='upper right'); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss magnitude")
    ax.set_title("Loss Component Decomposition")
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig10_loss_decomposition → {save_path}")


def fig11_per_class_recon(model, loader, cfg, save_path, n_per_class=5):
    """每个类的重建质量 (按类组织)"""
    model.eval()
    K = cfg.num_classes
    class_imgs = {k: [] for k in range(K)}

    with torch.no_grad():
        for y_img, y_label in loader:
            for k in range(K):
                mask = y_label == k
                if mask.sum() > 0 and len(class_imgs[k]) < n_per_class:
                    imgs = y_img[mask][:n_per_class - len(class_imgs[k])]
                    class_imgs[k].extend(imgs)
            if all(len(v) >= n_per_class for v in class_imgs.values()):
                break

    fig, axes = plt.subplots(K, n_per_class * 2, figsize=(n_per_class * 3, K * 1.3))
    with torch.no_grad():
        for k in range(K):
            imgs = torch.stack(class_imgs[k][:n_per_class]).to(cfg.device)
            mu, _ = model.enc(imgs)
            y_onehot = F.one_hot(torch.full((len(imgs),), k, device=cfg.device, dtype=torch.long),
                                 K).float()
            recon = model.dec(mu, y_onehot)
            for i in range(n_per_class):
                axes[k, i*2].imshow(imgs[i, 0].cpu(), cmap='gray'); axes[k, i*2].axis('off')
                axes[k, i*2+1].imshow(recon[i, 0].cpu(), cmap='gray'); axes[k, i*2+1].axis('off')
            axes[k, 0].set_ylabel(f"Class {k}", fontsize=8, rotation=0, labelpad=30)

    fig.suptitle("Per-Class Reconstruction (Orig → Recon)", fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig11_per_class_recon → {save_path}")


def fig12_gumbel_resp_heatmap(model, loader, cfg, save_path, n_samples=50):
    """Gumbel softmax resp 热力图"""
    model.eval()
    with torch.no_grad():
        y_img, y_label = next(iter(loader))
        y_img = y_img[:n_samples].to(cfg.device)
        y_label = y_label[:n_samples]
        _, info = model.forward_unlabeled(y_img, cfg)
        resp = info['resp'].cpu().numpy()  # [n_samples, K]

    # 按真实标签排序
    sort_idx = np.argsort(y_label.numpy())
    resp_sorted = resp[sort_idx]
    labels_sorted = y_label.numpy()[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(resp_sorted, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel("Cluster k"); ax.set_ylabel("Sample (sorted by true label)")
    ax.set_title("Gumbel Softmax Responsibilities")
    plt.colorbar(im, ax=ax, fraction=0.046, label="resp_k")

    # 标记类别边界
    prev = labels_sorted[0]
    for i, label in enumerate(labels_sorted):
        if label != prev:
            ax.axhline(y=i-0.5, color='white', linewidth=0.5)
            prev = label

    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig12_gumbel_resp_heatmap → {save_path}")


def fig13_interpolation(model, cfg, save_path, n_steps=10):
    """两类之间的 z 空间插值"""
    model.eval()
    K = cfg.num_classes
    n_pairs = min(5, K // 2)

    fig, axes = plt.subplots(n_pairs, n_steps, figsize=(n_steps * 1.2, n_pairs * 1.5))
    with torch.no_grad():
        for p in range(n_pairs):
            k1, k2 = p * 2, p * 2 + 1
            z1 = torch.randn(1, cfg.latent_dim).to(cfg.device)
            z2 = torch.randn(1, cfg.latent_dim).to(cfg.device)
            for s in range(n_steps):
                alpha = s / (n_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                y_onehot = F.one_hot(
                    torch.tensor([k1 if alpha < 0.5 else k2], device=cfg.device),
                    K).float()
                img = model.dec(z_interp, y_onehot)
                axes[p, s].imshow(img[0, 0].cpu(), cmap='gray')
                axes[p, s].axis('off')
            axes[p, 0].set_ylabel(f"{k1}→{k2}", fontsize=8, rotation=0, labelpad=25)

    fig.suptitle("Latent Space Interpolation", fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig13_interpolation → {save_path}")


def fig14_summary_table(logger, cfg, save_path, mode="unsup"):
    """训练总结表"""
    nmis = logger.records['nmi']
    accs = logger.records['posterior_acc']
    best_nmi = max(nmis)
    best_acc = max(accs)
    final_nmi = nmis[-1]
    final_acc = accs[-1]
    final_pi = logger.records['pi_values'][-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    table_data = [
        ["Mode", mode],
        ["Best NMI", f"{best_nmi:.4f}"],
        ["Best Posterior Acc", f"{best_acc:.4f}"],
        ["Final NMI", f"{final_nmi:.4f}"],
        ["Final Posterior Acc", f"{final_acc:.4f}"],
        ["Latent Dim", str(cfg.latent_dim)],
        ["β (KL weight)", f"{cfg.beta:.2f}"],
        ["Final τ", f"{cfg.current_gumbel_temp:.3f}"],
        ["Emission Model", "BCE (Bernoulli)" if cfg.use_bce else "MSE (Gaussian)"],
        ["π (min/max)", f"{min(final_pi):.3f} / {max(final_pi):.3f}"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c73d2'); cell.set_text_props(color='white', fontweight='bold')
    ax.set_title("Training Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig14_summary_table → {save_path}")


def fig15_x_conditionality(model, cfg, save_path, n_z=5):
    """
    ★ 关键诊断图: 固定 z, 改变 x → 观察生成图是否变化
    如果每列(同一z)下所有行(不同x)看起来一样 → x 被忽略了(坍缩)
    如果每列不同行有明显差异 → x 被正确使用
    """
    model.eval()
    K = cfg.num_classes

    fig, axes = plt.subplots(K, n_z, figsize=(n_z * 1.5, K * 1.3))
    with torch.no_grad():
        for j in range(n_z):
            z = torch.randn(1, cfg.latent_dim).to(cfg.device)
            for k in range(K):
                y_onehot = F.one_hot(
                    torch.tensor([k], device=cfg.device), K).float()
                img = model.dec(z, y_onehot)
                axes[k, j].imshow(img[0, 0].cpu(), cmap='gray')
                axes[k, j].axis('off')
            axes[0, j].set_title(f"z_{j}", fontsize=8)
        for k in range(K):
            axes[k, 0].set_ylabel(f"x={k}", fontsize=8, rotation=0, labelpad=20)

    xcond = measure_x_conditionality(model, cfg)
    fig.suptitle(f"x-Conditionality Check (xcond={xcond:.3f})\n"
                 f"Same z per column, different x per row",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig15_x_conditionality → {save_path}")


# ============================================================
# Generate ALL figures
# ============================================================
def generate_all_figures(model, logger, loader, cfg, mode="unsup"):
    fig_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("Generating all figures...")
    print("=" * 50)

    # 加载最佳模型
    best_path = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
    model.eval()

    fig01_training_curves(logger, os.path.join(fig_dir, "fig01_training_curves.png"), mode)
    fig02_nmi_acc_detail(logger, os.path.join(fig_dir, "fig02_nmi_acc_detail.png"))
    fig03_resp_entropy(logger, os.path.join(fig_dir, "fig03_resp_pi_entropy.png"))
    fig04_pi_evolution(logger, os.path.join(fig_dir, "fig04_pi_evolution.png"), cfg.num_classes)
    fig05_latent_space(model, loader, cfg, os.path.join(fig_dir, "fig05_latent_space_pca.png"), 'pca')
    fig06_generated_samples(model, cfg, os.path.join(fig_dir, "fig06_generated_samples.png"))
    fig07_confusion_matrix(model, loader, cfg, os.path.join(fig_dir, "fig07_confusion_matrix.png"))
    fig08_cluster_distribution(model, loader, cfg, os.path.join(fig_dir, "fig08_cluster_distribution.png"))
    fig09_reconstruction(model, loader, cfg, os.path.join(fig_dir, "fig09_reconstruction.png"))
    fig10_loss_decomposition(logger, os.path.join(fig_dir, "fig10_loss_decomposition.png"))
    fig11_per_class_recon(model, loader, cfg, os.path.join(fig_dir, "fig11_per_class_recon.png"))
    fig12_gumbel_resp_heatmap(model, loader, cfg, os.path.join(fig_dir, "fig12_resp_heatmap.png"))
    fig13_interpolation(model, cfg, os.path.join(fig_dir, "fig13_interpolation.png"))
    fig14_summary_table(logger, cfg, os.path.join(fig_dir, "fig14_summary_table.png"), mode)
    fig15_x_conditionality(model, cfg, os.path.join(fig_dir, "fig15_x_conditionality.png"))

    # 保存 logger
    logger.save(os.path.join(fig_dir, "training_log.json"))

    # ★ 最终 x-conditionality 得分
    xcond = measure_x_conditionality(model, cfg)
    print(f"\n★ Final x-conditionality score: {xcond:.4f}")
    if xcond < 0.3:
        print("  ⚠️  WARNING: Decoder largely ignores x! Consider reducing latent_dim further.")
    elif xcond > 0.6:
        print("  ✅  Decoder properly uses x as conditioning.")
    else:
        print("  🟡  Decoder partially uses x. May improve with more training.")

    print(f"\n✅ All 15 figures saved to {fig_dir}/")


# ============================================================
# Optuna 调参
# ============================================================
def run_optuna(mode, n_trials=15):
    import optuna

    def objective(trial):
        cfg = Config()
        cfg.latent_dim = trial.suggest_categorical("latent_dim", [2, 4, 8])  # ★ 不要 16/32
        cfg.beta = trial.suggest_float("beta", 0.1, 2.0)  # ★ 上限从 5→2
        cfg.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        cfg.init_gumbel_temp = trial.suggest_float("init_gumbel_temp", 0.2, 0.7)  # ★ 更低范围
        cfg.gumbel_anneal_rate = trial.suggest_float("gumbel_anneal_rate", 0.96, 0.995)  # ★ 新增
        cfg.current_gumbel_temp = cfg.init_gumbel_temp

        if mode == "semisup":
            cfg.alpha_unlabeled = trial.suggest_float("alpha_unlabeled", 0.5, 2.0)

        model = mVAE(cfg).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        if mode == "unsup":
            train_loader, val_loader = get_unsup_loaders(cfg)
            best_nmi = train_unsupervised(model, optimizer, train_loader, val_loader, cfg)
        else:
            lab, unlab, val = get_semisup_loaders(cfg)
            best_nmi = train_semisupervised(model, optimizer, lab, unlab, val, cfg)

        return -best_nmi

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"\nBest params: {study.best_params}")
    return study.best_params


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="mVAE Paper-Aligned Training")
    parser.add_argument("--mode", type=str, default="unsup", choices=["unsup", "semisup"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--labeled_per_class", type=int, default=100)
    parser.add_argument("--use_mse", action="store_true", help="Use MSE instead of BCE")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna first")
    parser.add_argument("--n_trials", type=int, default=15)
    args = parser.parse_args()

    cfg = Config()
    cfg.final_epochs = args.epochs
    cfg.latent_dim = args.latent_dim
    cfg.beta = args.beta
    cfg.lr = args.lr
    cfg.batch_size = args.batch_size
    cfg.labeled_per_class = args.labeled_per_class
    cfg.use_bce = not args.use_mse
    cfg.output_dir = f"./mVAE_{args.mode}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"mVAE Paper-Aligned Training")
    print(f"  Mode:       {args.mode}")
    print(f"  Latent dim: {cfg.latent_dim}")
    print(f"  β:          {cfg.beta}")
    print(f"  Emission:   {'BCE (Bernoulli)' if cfg.use_bce else 'MSE (Gaussian)'}")
    print(f"  Epochs:     {cfg.final_epochs}")
    print(f"  Device:     {cfg.device}")
    print("=" * 60)

    # --- Optuna ---
    if args.optuna:
        print("\n--- Running Optuna Hyperparameter Search ---")
        best_params = run_optuna(args.mode, args.n_trials)
        for k, v in best_params.items():
            setattr(cfg, k, v)
        cfg.current_gumbel_temp = cfg.init_gumbel_temp
        json.dump(best_params, open(os.path.join(cfg.output_dir, "best_params.json"), "w"), indent=2)
        print(f"Best params saved. Starting final training...\n")

    # --- Final Training ---
    logger = TrainingLogger()
    model = mVAE(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if args.mode == "unsup":
        train_loader, val_loader = get_unsup_loaders(cfg)
        best_nmi = train_unsupervised(model, optimizer, train_loader, val_loader,
                                       cfg, logger=logger, is_final=True)
        eval_loader = val_loader
    else:
        lab_loader, unlab_loader, val_loader = get_semisup_loaders(cfg)
        best_nmi = train_semisupervised(model, optimizer, lab_loader, unlab_loader,
                                         val_loader, cfg, logger=logger, is_final=True)
        eval_loader = val_loader

    print(f"\n✅ Training complete. Best NMI: {best_nmi:.4f}")

    # --- Generate all visualizations ---
    generate_all_figures(model, logger, eval_loader, cfg, mode=args.mode)

    # 保存配置
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
