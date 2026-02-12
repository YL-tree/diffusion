# mVAE_aligned.py
# ═══════════════════════════════════════════════════════════════
# Mixture VAE — v4: 严格论文公式, 从 epoch 1 开始, 不分阶段
#
# 论文公式 (Section 2.2):
#   -J^(t) = -Σ_k x_k log p(y|k,z,θ)     ① 加权重建
#            -Σ_k x_k log π_k              ② 先验
#            + KL(q(z|y)||p(z))             ③ KL
#            +Σ_k x_k log p(x=k|z,y,θ^(t)) ④ 后验校正
#
# 其中 x 通过 Gumbel softmax 从 p(x|z,y,θ^(t),Π^(t)) 采样
#
# ★★ 添加两个正则化 (不改变 ELBO 结构):
#   1. z_dropout: 训练时对 z 做 dropout, 迫使 decoder 不能只靠 z
#      理论依据: 等价于 p(z) 的噪声注入, 信息瓶颈正则化
#   2. balance_loss: Σ_k q̄_k log q̄_k (q̄ = batch 平均软分配)
#      理论依据: 等价于 Π 的 Dirichlet(1,...,1) 先验, 鼓励均匀使用所有类
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
# Model
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
        self.z_dropout_rate = cfg.z_dropout_rate

    @property
    def pi(self):
        return F.softmax(self.log_pi, dim=0)

    def forward_unlabeled(self, y, cfg):
        """
        论文 Section 2.2 的完整公式, 从 epoch 1 开始使用。

        唯一的两个额外正则化:
          - z_dropout: z 在进入 decoder 前被 dropout
          - balance_loss: 防止类坍缩
        """
        mu, logvar = self.enc(y)
        z_clean = reparameterize(mu, logvar)

        # ★★ z_dropout: 训练时随机丢弃 z 的维度
        # 迫使 decoder 不能仅靠 z 重建, 必须使用 x
        if self.training and self.z_dropout_rate > 0:
            z = F.dropout(z_clean, p=self.z_dropout_rate, training=True)
        else:
            z = z_clean

        B = y.size(0)

        # 每个类的 log p(y|x=k, z, θ) — z 已经 dropout 过
        recon_loglik, _ = compute_recon_loglik(
            self.dec, z, y, self.K, cfg.use_bce)  # [B, K]

        # E-step: 后验 p(x|z,y,θ^(t),Π^(t)) — 用 detached 参数
        log_pi = torch.log(self.pi + 1e-9)
        logits = log_pi.unsqueeze(0) + recon_loglik.detach()     # [B, K]
        log_posterior = F.log_softmax(logits, dim=1)

        # Gumbel softmax 采样 x
        resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp)  # [B, K]

        # ① 加权重建: -Σ_k x_k · log p(y|k,z,θ)
        weighted_recon = -(resp * recon_loglik).sum(dim=1).mean()

        # ② 先验: -Σ_k x_k · log π_k
        prior_loss = -(resp * log_pi.unsqueeze(0)).sum(dim=1).mean()

        # ③ KL(q(z|y) || p(z))
        kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # ④ 后验校正: +Σ_k x_k · log p(x=k|z,y,θ^(t))
        posterior_corr = (resp * log_posterior).sum(dim=1).mean()

        # 论文 ELBO
        loss = weighted_recon + cfg.beta * kl_z + prior_loss + posterior_corr

        # ★★ Balance loss (正则化, 非 ELBO 的一部分)
        # 防止所有样本被分到少数几个类
        soft_assign = F.softmax(recon_loglik.detach(), dim=1)    # [B, K]
        q_bar = soft_assign.mean(dim=0)                          # [K]
        balance_loss = torch.sum(q_bar * torch.log(q_bar + 1e-9))  # 负熵
        loss = loss + cfg.balance_weight * balance_loss

        resp_entropy = -(resp * torch.log(resp + 1e-9)).sum(dim=1).mean()

        return loss, {
            'recon': weighted_recon.item(), 'kl': kl_z.item(),
            'prior': prior_loss.item(), 'post_corr': posterior_corr.item(),
            'resp_ent': resp_entropy.item(), 'mu': mu.detach(),
            'resp': resp.detach(), 'balance': balance_loss.item(),
        }

    def forward_labeled(self, y, x_true, cfg):
        mu, logvar = self.enc(y)
        z_clean = reparameterize(mu, logvar)
        if self.training and self.z_dropout_rate > 0:
            z = F.dropout(z_clean, p=self.z_dropout_rate, training=True)
        else:
            z = z_clean
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
# Data
# ============================================================
def get_unsup_loaders(cfg):
    ds = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())
    n_train = int(0.9 * len(ds))
    train_set, val_set = random_split(ds, [n_train, len(ds) - n_train],
                                       generator=torch.Generator().manual_seed(42))
    return (DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True))

def get_semisup_loaders(cfg):
    ds = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())
    labels = np.array(ds.targets)
    labeled_idx, unlabeled_idx = [], []
    for c in range(cfg.num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.seed(42); np.random.shuffle(idx_c)
        labeled_idx.extend(idx_c[:cfg.labeled_per_class])
        unlabeled_idx.extend(idx_c[cfg.labeled_per_class:])
    return (
        DataLoader(Subset(ds, labeled_idx), batch_size=cfg.batch_size, shuffle=True),
        DataLoader(Subset(ds, unlabeled_idx), batch_size=cfg.batch_size, shuffle=True),
        DataLoader(Subset(ds, list(range(int(0.1*len(ds))))),
                   batch_size=cfg.batch_size, shuffle=False),
    )


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, cfg):
    model.eval()
    zs, ys_true, preds = [], [], []
    total_loss, n = 0, 0
    for y_img, y_label in loader:
        y_img = y_img.to(cfg.device)
        loss, info = model.forward_unlabeled(y_img, cfg)
        total_loss += loss.item(); n += 1
        zs.append(info['mu'].cpu().numpy())
        ys_true.append(y_label.numpy())
        if info.get('resp') is not None:
            preds.append(info['resp'].argmax(dim=1).cpu().numpy())
    zs = np.concatenate(zs); ys_true = np.concatenate(ys_true)
    nmi = compute_NMI(zs, ys_true, cfg.num_classes)
    post_acc = 0.0
    if preds:
        preds = np.concatenate(preds)
        post_acc, _ = compute_posterior_accuracy(preds, ys_true, cfg.num_classes)
    return nmi, post_acc, total_loss / max(n, 1)


@torch.no_grad()
def measure_x_conditionality(model, loader, cfg, n_z=50):
    model.eval()
    K = cfg.num_classes
    zs = []
    for y_img, _ in loader:
        mu, _ = model.enc(y_img.to(cfg.device))
        zs.append(mu)
        if sum(z.size(0) for z in zs) >= n_z: break
    z = torch.cat(zs)[:n_z]
    outputs = []
    for k in range(K):
        y_oh = F.one_hot(torch.full((n_z,), k, device=cfg.device, dtype=torch.long), K).float()
        outputs.append(model.dec(z, y_oh))
    outputs = torch.stack(outputs, dim=0)
    D = outputs[0].numel() // n_z
    flat = outputs.reshape(K, n_z, D)
    var_x = flat.var(dim=0).mean().item()
    var_z = flat.var(dim=1).mean().item()
    return var_x / (var_x + var_z + 1e-9)


# ============================================================
# Training: Unsupervised (单阶段, 论文公式)
# ============================================================
def train_unsupervised(model, optimizer, train_loader, val_loader, cfg,
                       logger=None, is_final=False):
    total_epochs = cfg.final_epochs if is_final else cfg.optuna_epochs
    best_nmi, best_acc = 0.0, 0.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        ep = {'recon': 0, 'kl': 0, 'prior': 0, 'post': 0, 'ent': 0, 'bal': 0}
        n_batches = 0

        # τ 退火
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

            ep['recon'] += info['recon']; ep['kl'] += info['kl']
            ep['prior'] += info['prior']; ep['post'] += info['post_corr']
            ep['ent'] += info['resp_ent']; ep['bal'] += info['balance']
            n_batches += 1

        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)
        if post_acc > best_acc:
            best_acc = post_acc
            if is_final:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
        best_nmi = max(best_nmi, nmi)

        pi_np = model.pi.detach().cpu().numpy()
        x_cond = 0.0
        if is_final and epoch % 10 == 0:
            x_cond = measure_x_conditionality(model, val_loader, cfg)

        if logger:
            logger.log(
                epoch=epoch, phase="Unsup",
                loss=val_loss,
                recon_loss=ep['recon']/n_batches, kl_loss=ep['kl']/n_batches,
                prior_loss=ep['prior']/n_batches, posterior_corr=ep['post']/n_batches,
                beta=cfg.beta, tau=cfg.current_gumbel_temp,
                nmi=nmi, posterior_acc=post_acc,
                resp_entropy=ep['ent']/n_batches,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                pi_values=pi_np, balance_loss=ep['bal']/n_batches,
            )

        if is_final:
            cond_str = f" xcond={x_cond:.3f}" if x_cond > 0 else ""
            print(f"  Ep {epoch:3d}/{total_epochs} "
                  f"| NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep['recon']/n_batches:.1f} KL={ep['kl']/n_batches:.2f} "
                  f"Bal={ep['bal']/n_batches:.3f} "
                  f"τ={cfg.current_gumbel_temp:.3f}{cond_str}")

    return best_nmi


# ============================================================
# Training: Semi-supervised
# ============================================================
def train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
                         val_loader, cfg, logger=None, is_final=False):
    total_epochs = cfg.final_epochs if is_final else cfg.optuna_epochs
    best_nmi, best_acc = 0.0, 0.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        ep = {'recon': 0, 'kl': 0, 'prior': 0, 'post': 0, 'ent': 0, 'bal': 0}
        n_batches = 0

        cfg.current_gumbel_temp = max(
            cfg.min_gumbel_temp,
            cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
            x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device)
            x_un = x_un.to(cfg.device)

            loss_lab, info_lab = model.forward_labeled(x_lab, y_lab, cfg)
            loss_un, info_un = model.forward_unlabeled(x_un, cfg)
            loss = loss_lab + cfg.alpha_unlabeled * loss_un

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            ep['recon'] += (info_lab['recon'] + info_un['recon']) / 2
            ep['kl'] += (info_lab['kl'] + info_un['kl']) / 2
            ep['prior'] += info_lab['prior']
            ep['post'] += info_un['post_corr']
            ep['ent'] += info_un['resp_ent']
            ep['bal'] += info_un['balance']
            n_batches += 1

        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)
        if post_acc > best_acc:
            best_acc = post_acc
            if is_final:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
        best_nmi = max(best_nmi, nmi)

        pi_np = model.pi.detach().cpu().numpy()
        if logger:
            logger.log(
                epoch=epoch, phase="SemiSup",
                loss=val_loss,
                recon_loss=ep['recon']/n_batches, kl_loss=ep['kl']/n_batches,
                prior_loss=ep['prior']/n_batches, posterior_corr=ep['post']/n_batches,
                beta=cfg.beta, tau=cfg.current_gumbel_temp,
                nmi=nmi, posterior_acc=post_acc,
                resp_entropy=ep['ent']/n_batches,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                pi_values=pi_np, balance_loss=ep['bal']/n_batches,
            )
        if is_final:
            print(f"  Ep {epoch:3d}/{total_epochs} "
                  f"| NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep['recon']/n_batches:.1f} KL={ep['kl']/n_batches:.2f}")

    return best_nmi


# ============================================================
# Visualization
# ============================================================
COLORS = ['#2c73d2', '#ff6b6b', '#51cf66', '#ffa94d', '#845ef7',
          '#f06595', '#20c997', '#fab005', '#339af0', '#ff8787']
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def fig01_training_curves(logger, save_path, mode="unsup"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs = logger.records['epoch']

    ax = axes[0, 0]
    ax.plot(epochs, logger.records['loss'], color=COLORS[0], linewidth=1.5)
    ax.set_title("Validation Loss"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, logger.records['recon_loss'], label='Recon', color=COLORS[0])
    ax.plot(epochs, logger.records['kl_loss'], label='KL', color=COLORS[1])
    bal = logger.records.get('balance_loss', [])
    if bal and any(v and v != 0 for v in bal):
        ax.plot(epochs, [v if v else 0 for v in bal], label='Balance', color=COLORS[2])
    ax.legend(fontsize=9); ax.set_title("Loss Components"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, logger.records['nmi'], label='NMI', color=COLORS[0], linewidth=2)
    ax.plot(epochs, logger.records['posterior_acc'], label='Post.Acc',
            color=COLORS[1], linewidth=2, linestyle='--')
    ax.legend(fontsize=10); ax.set_title("Clustering Quality")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, logger.records['tau'], label='τ', color=COLORS[4])
    ax.legend(); ax.set_title("Gumbel τ"); ax.grid(alpha=0.3)

    fig.suptitle(f"mVAE Training ({mode})", fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig01 → {save_path}")


def fig06_generated_samples(model, cfg, save_path, n_per_class=10):
    model.eval()
    z = torch.randn(n_per_class, cfg.latent_dim).to(cfg.device)
    all_samples = []
    with torch.no_grad():
        for k in range(cfg.num_classes):
            y_oh = F.one_hot(torch.full((n_per_class,), k, device=cfg.device,
                                         dtype=torch.long), cfg.num_classes).float()
            all_samples.append(model.dec(z, y_oh))
    save_image(torch.cat(all_samples), save_path, nrow=n_per_class, normalize=True)
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
                    class_imgs[k].extend(y_img[mask][:n_per_class - len(class_imgs[k])])
            if all(len(v) >= n_per_class for v in class_imgs.values()): break

    label_to_cluster = None
    if mode == "unsup":
        all_preds, all_labels = [], []
        with torch.no_grad():
            for y_img, y_label in loader:
                _, info = model.forward_unlabeled(y_img.to(cfg.device), cfg)
                if info.get('resp') is not None:
                    all_preds.append(info['resp'].argmax(dim=1).cpu().numpy())
                    all_labels.append(y_label.numpy())
                if sum(p.shape[0] for p in all_preds) > 2000: break
        if all_preds:
            _, mapping = compute_posterior_accuracy(
                np.concatenate(all_preds), np.concatenate(all_labels), K)
            label_to_cluster = {v: k for k, v in mapping.items()}

    fig, axes = plt.subplots(K, n_per_class*2, figsize=(n_per_class*3, K*1.3))
    with torch.no_grad():
        for k in range(K):
            imgs = torch.stack(class_imgs[k][:n_per_class]).to(cfg.device)
            mu, _ = model.enc(imgs)
            ck = label_to_cluster.get(k, k) if label_to_cluster else k
            y_oh = F.one_hot(torch.full((len(imgs),), ck, device=cfg.device,
                                         dtype=torch.long), K).float()
            recon = model.dec(mu, y_oh)
            for i in range(n_per_class):
                axes[k, i*2].imshow(imgs[i,0].cpu(), cmap='gray'); axes[k, i*2].axis('off')
                axes[k, i*2+1].imshow(recon[i,0].cpu(), cmap='gray'); axes[k, i*2+1].axis('off')
            axes[k, 0].set_ylabel(f"{k}", fontsize=8, rotation=0, labelpad=15)
    fig.suptitle("Per-Class Recon (Orig → Recon)", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig11 → {save_path}")


def fig15_x_conditionality(model, loader, cfg, save_path, n_z=5):
    model.eval()
    K = cfg.num_classes
    with torch.no_grad():
        y_img, _ = next(iter(loader))
        mu, _ = model.enc(y_img[:n_z].to(cfg.device))
    fig, axes = plt.subplots(K, n_z, figsize=(n_z*1.5, K*1.3))
    with torch.no_grad():
        for j in range(n_z):
            z = mu[j:j+1]
            for k in range(K):
                y_oh = F.one_hot(torch.tensor([k], device=cfg.device), K).float()
                img = model.dec(z, y_oh)
                axes[k,j].imshow(img[0,0].cpu(), cmap='gray'); axes[k,j].axis('off')
            axes[0,j].set_title(f"z_{j}", fontsize=8)
        for k in range(K):
            axes[k,0].set_ylabel(f"x={k}", fontsize=8, rotation=0, labelpad=20)
    xcond = measure_x_conditionality(model, loader, cfg)
    fig.suptitle(f"x-Conditionality (xcond={xcond:.3f})", fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig15 → {save_path}")


def generate_all_figures(model, logger, loader, cfg, mode="unsup"):
    fig_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    best_path = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
    model.eval()
    print("\n" + "="*50 + "\nGenerating figures...\n" + "="*50)
    fig01_training_curves(logger, os.path.join(fig_dir, "fig01_training_curves.png"), mode)
    fig06_generated_samples(model, cfg, os.path.join(fig_dir, "fig06_generated_samples.png"))
    fig11_per_class_recon(model, loader, cfg, os.path.join(fig_dir, "fig11_per_class_recon.png"), mode=mode)
    fig15_x_conditionality(model, loader, cfg, os.path.join(fig_dir, "fig15_x_conditionality.png"))
    logger.save(os.path.join(fig_dir, "training_log.json"))
    xcond = measure_x_conditionality(model, loader, cfg)
    print(f"\n★ Final x-conditionality: {xcond:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="mVAE v4")
    parser.add_argument("--mode", default="unsup", choices=["unsup", "semisup"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--z_dropout", type=float, default=0.5)
    parser.add_argument("--balance_weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--decoder", default="film", choices=["film", "concat"])
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--tau_rate", type=float, default=0.98)
    parser.add_argument("--labeled_per_class", type=int, default=100)
    args = parser.parse_args()

    cfg = Config()
    cfg.final_epochs = args.epochs
    cfg.latent_dim = args.latent_dim
    cfg.beta = args.beta
    cfg.z_dropout_rate = args.z_dropout
    cfg.balance_weight = args.balance_weight
    cfg.lr = args.lr
    cfg.batch_size = args.batch_size
    cfg.decoder_type = args.decoder
    cfg.init_gumbel_temp = args.tau_start
    cfg.min_gumbel_temp = args.tau_min
    cfg.gumbel_anneal_rate = args.tau_rate
    cfg.current_gumbel_temp = cfg.init_gumbel_temp
    cfg.labeled_per_class = args.labeled_per_class
    cfg.output_dir = f"./mVAE_{args.mode}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"mVAE v4 — Paper formula + z_dropout + balance_loss")
    print(f"  Mode:          {args.mode}")
    print(f"  Decoder:       {cfg.decoder_type}")
    print(f"  Latent dim:    {cfg.latent_dim}")
    print(f"  β:             {cfg.beta}")
    print(f"  z_dropout:     {cfg.z_dropout_rate}")
    print(f"  balance_w:     {cfg.balance_weight}")
    print(f"  τ:             {cfg.init_gumbel_temp} → {cfg.min_gumbel_temp} (rate={cfg.gumbel_anneal_rate})")
    print(f"  Epochs:        {cfg.final_epochs}")
    print(f"  Device:        {cfg.device}")
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
        lab, unlab, val = get_semisup_loaders(cfg)
        best_nmi = train_semisupervised(model, optimizer, lab, unlab, val,
                                         cfg, logger=logger, is_final=True)
        eval_loader = val

    print(f"\n✅ Done. Best NMI: {best_nmi:.4f}")
    generate_all_figures(model, logger, eval_loader, cfg, mode=args.mode)
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
