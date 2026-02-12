# mVAE_aligned.py
# ═══════════════════════════════════════════════════════════════
# Mixture VAE — v3: 借鉴 HMM-VAE 的两阶段训练策略
#
# ★★ 核心设计 (与 HMM-VAE 完全对齐):
#   Phase 1 (Pretrain, 前 30 epoch):
#     - argmax 硬分配 (不走后验采样)
#     - z_dropout=0.7 (压制 z, 迫使 decoder 依赖 x)
#     - β=50.0 (KL 强惩罚)
#     - balance_loss (防止类坍缩, 鼓励均匀分配)
#     - Π 不参与优化
#     → 目标: decoder 学会 "不同 x 生成不同数字"
#
#   Phase 2 (EM, 后 70 epoch):
#     - 切换到论文的 partially variational EM
#     - z_dropout 关闭, β 降到正常值
#     - Gumbel softmax 从后验采样
#     - Π 开始学习
#     → 目标: 在 decoder 已专门化的基础上, 学习正确的聚类
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
    Config, Encoder, ConditionalDecoder, FiLMDecoder,
    reparameterize, gumbel_softmax_sample, compute_recon_loglik,
    compute_NMI, compute_posterior_accuracy, TrainingLogger
)

# ============================================================
# Model: mVAE (v3)
# ============================================================
class mVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(cfg.latent_dim)
        if getattr(cfg, 'decoder_type', 'concat') == 'film':
            self.dec = FiLMDecoder(cfg.latent_dim, cfg.num_classes, cfg.hidden_dim)
        else:
            self.dec = ConditionalDecoder(cfg.latent_dim, cfg.num_classes, cfg.hidden_dim)
        self.log_pi = nn.Parameter(torch.zeros(cfg.num_classes))
        self.K = cfg.num_classes

        # ★★ v3: z_dropout 控制
        self.enable_z_dropout = False
        self.z_dropout_rate = 0.7

    @property
    def pi(self):
        return F.softmax(self.log_pi, dim=0)

    def _encode_and_sample(self, y, cfg):
        """编码 + 重参数化 + 可选 z_dropout"""
        mu, logvar = self.enc(y)
        z = reparameterize(mu, logvar)
        # ★★ v3: z_dropout (训练时, Phase 1 启用)
        if self.training and self.enable_z_dropout:
            z = F.dropout(z, p=self.z_dropout_rate)
        return mu, logvar, z

    # ========================================================
    # ★★ Phase 1: Pretrain (emission bootstrap)
    #    跟 HMM-VAE 完全一致的逻辑
    # ========================================================
    def forward_pretrain(self, y, cfg):
        """
        Phase 1: argmax 硬分配, 不走后验采样

        等价于 HMM-VAE 的:
          best_k = log_bk.argmax(dim=2)
          X_d = F.one_hot(best_k, K).float()
          emission_recon = -torch.sum(X_d * log_bk) / (B * T)
          balance_loss = torch.sum(q * torch.log(q + 1e-9))
        """
        mu, logvar, z = self._encode_and_sample(y, cfg)
        B = y.size(0)

        # 对每个类计算 log p(y|x=k, z, θ)
        recon_loglik, _ = compute_recon_loglik(
            self.dec, z, y, self.K, cfg.use_bce)  # [B, K]

        # ★★ argmax 硬分配 (detached, 不传梯度到分配过程)
        with torch.no_grad():
            best_k = recon_loglik.argmax(dim=1)                  # [B]
            X_d = F.one_hot(best_k, self.K).float()              # [B, K]

        # ★★ 加权重建 (梯度只传到 decoder, 不传到分配)
        emission_recon = -(X_d * recon_loglik).sum(dim=1).mean()

        # ★★ KL 散度 (高 β 压制 z)
        kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # ★★ Balance loss: 防止所有样本被分到同一个类
        # softmax 得到软分配, 然后看平均分配是否均匀
        soft_assign = F.softmax(recon_loglik, dim=1)             # [B, K]
        q = soft_assign.mean(dim=0)                              # [K], 平均分配概率
        balance_loss = torch.sum(q * torch.log(q + 1e-9))       # 负熵, 越均匀越小

        loss = emission_recon + cfg.pretrain_beta * kl_z + cfg.balance_weight * balance_loss

        # 诊断
        resp_entropy = -(X_d * torch.log(X_d + 1e-9)).sum(dim=1).mean()  # 硬分配, 熵=0

        return loss, {
            'recon': emission_recon.item(), 'kl': kl_z.item(),
            'prior': 0.0, 'post_corr': 0.0,
            'resp_ent': resp_entropy.item(), 'mu': mu.detach(),
            'resp': X_d.detach(),
            'balance': balance_loss.item(),
        }

    # ========================================================
    # ★★ Phase 2: 论文的 partially variational EM
    #    (在 decoder 已专门化的基础上运行)
    # ========================================================
    def forward_unlabeled(self, y, cfg, epoch=None):
        """
        Phase 2: 论文公式 — Gumbel softmax 从后验采样
        """
        mu, logvar, z = self._encode_and_sample(y, cfg)
        B = y.size(0)

        recon_loglik, _ = compute_recon_loglik(
            self.dec, z, y, self.K, cfg.use_bce)

        # E-step: 后验
        log_pi = torch.log(self.pi + 1e-9)
        logits = log_pi.unsqueeze(0) + recon_loglik.detach()
        log_posterior = F.log_softmax(logits, dim=1)

        # Gumbel softmax 采样
        resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp, hard=False)

        # 论文 -J^(t) 各项
        weighted_recon = -(resp * recon_loglik).sum(dim=1).mean()
        prior_loss = -(resp * log_pi.unsqueeze(0)).sum(dim=1).mean()
        kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        posterior_corr = (resp * log_posterior).sum(dim=1).mean()

        loss = weighted_recon + cfg.beta * kl_z + prior_loss + posterior_corr

        resp_entropy = -(resp * torch.log(resp + 1e-9)).sum(dim=1).mean()

        return loss, {
            'recon': weighted_recon.item(), 'kl': kl_z.item(),
            'prior': prior_loss.item(), 'post_corr': posterior_corr.item(),
            'resp_ent': resp_entropy.item(), 'mu': mu.detach(),
            'resp': resp.detach(), 'balance': 0.0,
        }

    def forward_labeled(self, y, x_true, cfg):
        mu, logvar, z = self._encode_and_sample(y, cfg)
        B = y.size(0)

        x_onehot = F.one_hot(x_true, self.K).float()
        y_recon = self.dec(z, x_onehot)

        if cfg.use_bce:
            recon_loss = F.binary_cross_entropy(y_recon, y, reduction='sum') / B
        else:
            recon_loss = F.mse_loss(y_recon, y, reduction='sum') / B

        kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        log_pi = torch.log(self.pi + 1e-9)
        prior_loss = -log_pi[x_true].mean()

        loss = recon_loss + cfg.beta * kl_z + prior_loss
        return loss, {
            'recon': recon_loss.item(), 'kl': kl_z.item(),
            'prior': prior_loss.item(), 'post_corr': 0.0,
            'resp_ent': 0.0, 'mu': mu.detach(), 'balance': 0.0,
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
    model.eval()
    zs, ys_true, preds = [], [], []
    total_loss = 0
    n_batches = 0

    for y_img, y_label in loader:
        y_img = y_img.to(cfg.device)
        loss, info = model.forward_unlabeled(y_img, cfg, epoch=999)
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


@torch.no_grad()
def measure_x_conditionality(model, loader, cfg, n_z=50):
    model.eval()
    K = cfg.num_classes
    zs = []
    for y_img, _ in loader:
        y_img = y_img.to(cfg.device)
        mu, _ = model.enc(y_img)
        zs.append(mu)
        if sum(z.size(0) for z in zs) >= n_z:
            break
    z = torch.cat(zs)[:n_z]

    outputs = []
    for k in range(K):
        y_onehot = F.one_hot(
            torch.full((n_z,), k, device=cfg.device, dtype=torch.long), K).float()
        out = model.dec(z, y_onehot)
        outputs.append(out)
    outputs = torch.stack(outputs, dim=0)
    D = outputs[0].numel() // n_z
    flat = outputs.reshape(K, n_z, D)
    var_x = flat.var(dim=0).mean().item()
    var_z = flat.var(dim=1).mean().item()
    return var_x / (var_x + var_z + 1e-9)


# ============================================================
# Training: Unsupervised (两阶段)
# ============================================================
def train_unsupervised(model, optimizer, train_loader, val_loader, cfg,
                       logger=None, is_final=False):
    total_epochs = cfg.final_epochs if is_final else cfg.optuna_epochs
    best_nmi = 0.0
    best_acc = 0.0
    pretrain_epochs = getattr(cfg, 'pretrain_epochs', 30)

    for epoch in range(1, total_epochs + 1):
        model.train()
        ep_recon, ep_kl, ep_prior, ep_post, ep_ent, ep_bal = 0, 0, 0, 0, 0, 0
        n_batches = 0

        # ========================================
        # ★★ 两阶段切换 (核心!)
        # ========================================
        if epoch <= pretrain_epochs:
            # Phase 1: Emission Bootstrap
            phase = "Pretrain"
            model.enable_z_dropout = True
            model.z_dropout_rate = cfg.z_dropout_rate
            current_beta = cfg.pretrain_beta
            # Π 不参与优化
            model.log_pi.requires_grad = False
        else:
            # Phase 2: Partially Variational EM
            phase = "EM"
            model.enable_z_dropout = False
            current_beta = cfg.beta
            model.log_pi.requires_grad = True

            # Gumbel τ 退火 (只在 Phase 2)
            cfg.current_gumbel_temp = max(
                cfg.min_gumbel_temp,
                cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        # ★★ Phase 2 使用不同学习率
        if epoch == pretrain_epochs + 1 and is_final:
            print(f"\n{'='*50}")
            print(f"  Phase 2 starts! Switching to EM mode.")
            print(f"  z_dropout=OFF, β={cfg.beta}, τ starts annealing")
            print(f"{'='*50}\n")
            # 重建 optimizer, 给 Π 更高学习率
            vae_params = [p for n, p in model.named_parameters()
                          if 'log_pi' not in n]
            optimizer = torch.optim.Adam([
                {'params': vae_params, 'lr': cfg.finetune_lr},
                {'params': [model.log_pi], 'lr': cfg.pi_lr},
            ])

        for y_img, _ in train_loader:
            y_img = y_img.to(cfg.device)

            if epoch <= pretrain_epochs:
                # ★★ Phase 1: pretrain
                loss, info = model.forward_pretrain(y_img, cfg)
            else:
                # ★★ Phase 2: 论文 EM
                loss, info = model.forward_unlabeled(y_img, cfg, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            ep_recon += info['recon']
            ep_kl += info['kl']
            ep_prior += info.get('prior', 0)
            ep_post += info.get('post_corr', 0)
            ep_ent += info.get('resp_ent', 0)
            ep_bal += info.get('balance', 0)
            n_batches += 1

        # Evaluate (always use EM mode for consistency)
        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)

        if post_acc > best_acc:
            best_acc = post_acc
            if is_final:
                torch.save(model.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
        best_nmi = max(best_nmi, nmi)

        pi_np = model.pi.detach().cpu().numpy()

        x_cond = 0.0
        if is_final and epoch % 10 == 0:
            x_cond = measure_x_conditionality(model, val_loader, cfg)

        if logger:
            logger.log(
                epoch=epoch, phase=phase,
                loss=val_loss,
                recon_loss=ep_recon / n_batches,
                kl_loss=ep_kl / n_batches,
                prior_loss=ep_prior / n_batches,
                posterior_corr=ep_post / n_batches,
                beta=current_beta if epoch <= pretrain_epochs else cfg.beta,
                tau=cfg.current_gumbel_temp,
                nmi=nmi, posterior_acc=post_acc,
                resp_entropy=ep_ent / n_batches,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                pi_values=pi_np,
                balance_loss=ep_bal / n_batches,
            )

        if is_final:
            cond_str = f" xcond={x_cond:.3f}" if x_cond > 0 else ""
            bal_str = f" Bal={ep_bal/n_batches:.2f}" if epoch <= pretrain_epochs else ""
            print(f"  Ep {epoch:3d}/{total_epochs} [{phase:8s}] "
                  f"| NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep_recon/n_batches:.1f} KL={ep_kl/n_batches:.2f} "
                  f"β={current_beta if epoch<=pretrain_epochs else cfg.beta:.1f} "
                  f"τ={cfg.current_gumbel_temp:.3f}"
                  f"{bal_str}{cond_str}")

    return best_nmi


# ============================================================
# Training: Semi-supervised (两阶段)
# ============================================================
def train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
                         val_loader, cfg, logger=None, is_final=False):
    total_epochs = cfg.final_epochs if is_final else cfg.optuna_epochs
    best_nmi = 0.0
    best_acc = 0.0
    pretrain_epochs = getattr(cfg, 'pretrain_epochs', 30)

    for epoch in range(1, total_epochs + 1):
        model.train()
        ep_recon, ep_kl, ep_prior, ep_post, ep_ent, ep_bal = 0, 0, 0, 0, 0, 0
        n_batches = 0

        if epoch <= pretrain_epochs:
            phase = "Pretrain"
            model.enable_z_dropout = True
            model.z_dropout_rate = cfg.z_dropout_rate
            model.log_pi.requires_grad = False
        else:
            phase = "EM"
            model.enable_z_dropout = False
            model.log_pi.requires_grad = True
            cfg.current_gumbel_temp = max(
                cfg.min_gumbel_temp,
                cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        if epoch == pretrain_epochs + 1 and is_final:
            print(f"\n{'='*50}")
            print(f"  Phase 2 starts! Switching to EM mode.")
            print(f"{'='*50}\n")
            vae_params = [p for n, p in model.named_parameters()
                          if 'log_pi' not in n]
            optimizer = torch.optim.Adam([
                {'params': vae_params, 'lr': cfg.finetune_lr},
                {'params': [model.log_pi], 'lr': cfg.pi_lr},
            ])

        for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
            x_lab = x_lab.to(cfg.device)
            y_lab = y_lab.to(cfg.device)
            x_un = x_un.to(cfg.device)

            loss_lab, info_lab = model.forward_labeled(x_lab, y_lab, cfg)

            if epoch <= pretrain_epochs:
                loss_un, info_un = model.forward_pretrain(x_un, cfg)
            else:
                loss_un, info_un = model.forward_unlabeled(x_un, cfg, epoch=epoch)

            loss = loss_lab + cfg.alpha_unlabeled * loss_un

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            ep_recon += (info_lab['recon'] + info_un['recon']) / 2
            ep_kl += (info_lab['kl'] + info_un['kl']) / 2
            ep_prior += info_lab.get('prior', 0)
            ep_post += info_un.get('post_corr', 0)
            ep_ent += info_un.get('resp_ent', 0)
            ep_bal += info_un.get('balance', 0)
            n_batches += 1

        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)

        if post_acc > best_acc:
            best_acc = post_acc
            if is_final:
                torch.save(model.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
        best_nmi = max(best_nmi, nmi)

        pi_np = model.pi.detach().cpu().numpy()

        if logger:
            logger.log(
                epoch=epoch, phase=phase,
                loss=val_loss,
                recon_loss=ep_recon / n_batches,
                kl_loss=ep_kl / n_batches,
                prior_loss=ep_prior / n_batches,
                posterior_corr=ep_post / n_batches,
                beta=cfg.pretrain_beta if epoch <= pretrain_epochs else cfg.beta,
                tau=cfg.current_gumbel_temp,
                nmi=nmi, posterior_acc=post_acc,
                resp_entropy=ep_ent / n_batches,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                pi_values=pi_np,
                balance_loss=ep_bal / n_batches,
            )

        if is_final:
            print(f"  Ep {epoch:3d}/{total_epochs} [{phase:8s}] "
                  f"| NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep_recon/n_batches:.1f} KL={ep_kl/n_batches:.2f}")

    return best_nmi


# ============================================================
# Visualization (与之前一致, 省略部分简单图)
# ============================================================
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs = logger.records['epoch']

    # 找 phase 边界
    phases = logger.records['phase']
    pretrain_end = None
    for i, p in enumerate(phases):
        if p == 'EM' and pretrain_end is None:
            pretrain_end = epochs[i]
            break

    ax = axes[0, 0]
    ax.plot(epochs, logger.records['loss'], color=COLORS[0], linewidth=1.5)
    if pretrain_end: ax.axvline(x=pretrain_end, color='red', linestyle='--', alpha=0.5, label='Phase 2 start')
    ax.set_title("Validation Loss"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    if pretrain_end: ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(epochs, logger.records['recon_loss'], label='Recon', color=COLORS[0])
    ax.plot(epochs, logger.records['kl_loss'], label='KL', color=COLORS[1])
    bal = logger.records.get('balance_loss', [])
    if bal and any(v and v != 0 for v in bal):
        ax.plot(epochs, [v if v else 0 for v in bal], label='Balance', color=COLORS[2])
    if pretrain_end: ax.axvline(x=pretrain_end, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9); ax.set_title("Loss Components"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, logger.records['nmi'], label='NMI', color=COLORS[0], linewidth=2)
    ax.plot(epochs, logger.records['posterior_acc'], label='Post.Acc', color=COLORS[1],
            linewidth=2, linestyle='--')
    if pretrain_end: ax.axvline(x=pretrain_end, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10); ax.set_title("Clustering Quality")
    ax.set_xlabel("Epoch"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, logger.records['tau'], label='τ (Gumbel)', color=COLORS[4])
    ax2 = ax.twinx()
    ax2.plot(epochs, logger.records['beta'], label='β (KL)', color=COLORS[3], linestyle='--')
    if pretrain_end: ax.axvline(x=pretrain_end, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("τ"); ax2.set_ylabel("β")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.set_title("Schedule: τ & β"); ax.grid(alpha=0.3)

    fig.suptitle(f"mVAE Training ({mode}) — Two-Phase", fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig01 → {save_path}")


def fig06_generated_samples(model, cfg, save_path, n_per_class=10):
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
    print(f"  ✓ fig06 → {save_path}")


def fig11_per_class_recon(model, loader, cfg, save_path, n_per_class=5, mode="unsup"):
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

    label_to_cluster = None
    if mode == "unsup":
        all_preds, all_labels = [], []
        with torch.no_grad():
            for y_img, y_label in loader:
                y_img = y_img.to(cfg.device)
                _, info = model.forward_unlabeled(y_img, cfg, epoch=999)
                if 'resp' in info and info['resp'] is not None:
                    all_preds.append(info['resp'].argmax(dim=1).cpu().numpy())
                    all_labels.append(y_label.numpy())
                if sum(p.shape[0] for p in all_preds) > 2000:
                    break
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            _, mapping = compute_posterior_accuracy(all_preds, all_labels, K)
            label_to_cluster = {v: k for k, v in mapping.items()}

    fig, axes = plt.subplots(K, n_per_class * 2, figsize=(n_per_class * 3, K * 1.3))
    with torch.no_grad():
        for k in range(K):
            imgs = torch.stack(class_imgs[k][:n_per_class]).to(cfg.device)
            mu, _ = model.enc(imgs)
            cluster_k = label_to_cluster.get(k, k) if label_to_cluster else k
            y_onehot = F.one_hot(
                torch.full((len(imgs),), cluster_k, device=cfg.device, dtype=torch.long), K).float()
            recon = model.dec(mu, y_onehot)
            for i in range(n_per_class):
                axes[k, i*2].imshow(imgs[i, 0].cpu(), cmap='gray'); axes[k, i*2].axis('off')
                axes[k, i*2+1].imshow(recon[i, 0].cpu(), cmap='gray'); axes[k, i*2+1].axis('off')
            axes[k, 0].set_ylabel(f"Digit {k}", fontsize=7, rotation=0, labelpad=30)
    fig.suptitle("Per-Class Reconstruction", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig11 → {save_path}")


def fig15_x_conditionality(model, loader, cfg, save_path, n_z=5):
    model.eval()
    K = cfg.num_classes
    with torch.no_grad():
        y_img, _ = next(iter(loader))
        y_img = y_img[:n_z].to(cfg.device)
        mu, _ = model.enc(y_img)
        z_real = mu

    fig, axes = plt.subplots(K, n_z, figsize=(n_z * 1.5, K * 1.3))
    with torch.no_grad():
        for j in range(n_z):
            z = z_real[j:j+1]
            for k in range(K):
                y_onehot = F.one_hot(torch.tensor([k], device=cfg.device), K).float()
                img = model.dec(z, y_onehot)
                axes[k, j].imshow(img[0, 0].cpu(), cmap='gray')
                axes[k, j].axis('off')
            axes[0, j].set_title(f"z_{j}", fontsize=8)
        for k in range(K):
            axes[k, 0].set_ylabel(f"x={k}", fontsize=8, rotation=0, labelpad=20)

    xcond = measure_x_conditionality(model, loader, cfg)
    fig.suptitle(f"x-Conditionality (xcond={xcond:.3f})", fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig15 → {save_path}")


def generate_all_figures(model, logger, loader, cfg, mode="unsup"):
    fig_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("Generating figures...")
    print("=" * 50)

    best_path = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
    model.eval()

    fig01_training_curves(logger, os.path.join(fig_dir, "fig01_training_curves.png"), mode)
    fig06_generated_samples(model, cfg, os.path.join(fig_dir, "fig06_generated_samples.png"))
    fig11_per_class_recon(model, loader, cfg, os.path.join(fig_dir, "fig11_per_class_recon.png"), mode=mode)
    fig15_x_conditionality(model, loader, cfg, os.path.join(fig_dir, "fig15_x_conditionality.png"))
    logger.save(os.path.join(fig_dir, "training_log.json"))

    xcond = measure_x_conditionality(model, loader, cfg)
    print(f"\n★ Final x-conditionality: {xcond:.4f}")
    print(f"✅ Figures saved to {fig_dir}/")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="mVAE v3 — Two-Phase Training")
    parser.add_argument("--mode", type=str, default="unsup", choices=["unsup", "semisup"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--pretrain_beta", type=float, default=50.0)
    parser.add_argument("--z_dropout", type=float, default=0.7)
    parser.add_argument("--balance_weight", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--decoder", type=str, default="film", choices=["film", "concat"])
    parser.add_argument("--labeled_per_class", type=int, default=100)
    args = parser.parse_args()

    cfg = Config()
    cfg.final_epochs = args.epochs
    cfg.latent_dim = args.latent_dim
    cfg.pretrain_epochs = args.pretrain_epochs
    cfg.pretrain_beta = args.pretrain_beta
    cfg.z_dropout_rate = args.z_dropout
    cfg.balance_weight = args.balance_weight
    cfg.beta = args.beta
    cfg.lr = args.lr
    cfg.finetune_lr = args.finetune_lr
    cfg.batch_size = args.batch_size
    cfg.decoder_type = args.decoder
    cfg.labeled_per_class = args.labeled_per_class
    cfg.output_dir = f"./mVAE_{args.mode}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"mVAE v3 — Two-Phase Training (borrowing from HMM-VAE)")
    print(f"  Mode:             {args.mode}")
    print(f"  Decoder:          {cfg.decoder_type}")
    print(f"  Latent dim:       {cfg.latent_dim}")
    print(f"  Phase 1 (Pretrain): {cfg.pretrain_epochs} epochs")
    print(f"    β={cfg.pretrain_beta}, z_dropout={cfg.z_dropout_rate}, balance_w={cfg.balance_weight}")
    print(f"  Phase 2 (EM):     {cfg.final_epochs - cfg.pretrain_epochs} epochs")
    print(f"    β={cfg.beta}, τ: {cfg.init_gumbel_temp}→{cfg.min_gumbel_temp}")
    print(f"  Device:           {cfg.device}")
    print("=" * 60)

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
    generate_all_figures(model, logger, eval_loader, cfg, mode=args.mode)

    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
