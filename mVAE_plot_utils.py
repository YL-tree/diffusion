"""
mVAE_plot_utils.py — mVAE 论文级可视化工具
============================================

参考 HMM-VAE 的 plot_utils.py, 为 mVAE (Section 2.2) 定制.

图表清单 (共 12 张):
  fig01  Training Curves       — Loss 分量 / NMI+Acc / Pi Entropy / τ
  fig02  Confusion Matrix      — 后验预测 vs 真实标签 (Hungarian 对齐)
  fig03  Conditional Gen Grid  — 行=x (类别), 列=不同 z 采样
  fig04  Latent Space           — t-SNE 或 2D scatter (z_dim=2)
  fig05  Class Frequency        — π 分布 (vs uniform)
  fig06  Per-Class Recon        — 原图 → 重构对比
  fig07  x-Conditionality Grid — 固定 z, 变 x 的生成效果
  fig08  Pi Evolution           — π 各分量随 epoch 变化
  fig09  Resp Entropy & τ      — 后验 sharpness 与温度退火
  fig10  Recon Samples          — 随机样本的输入 vs 重构
  fig11  Per-Digit Accuracy     — 每个数字的后验准确率
  fig12  Balance Loss Effect    — balance loss 与 π 均衡性

用法:
  from mVAE_plot_utils import generate_all_mVAE_figures

依赖: mVAE_aligned.py, mVAE_common.py
"""

import os, json
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ============================================================
# 全局样式
# ============================================================
STYLE = {
    "fig_dpi": 200,
    "font_title": 15,
    "font_label": 12,
    "font_tick": 10,
    "font_legend": 9,
    "color_primary": "#2196F3",
    "color_secondary": "#4CAF50",
    "color_tertiary": "#FF9800",
    "color_accent": "#E91E63",
    "color_neutral": "#9E9E9E",
    "linewidth": 1.8,
}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def _tab10(k, K=10):
    return plt.cm.tab10(k / K)


# ============================================================
# 数据收集辅助
# ============================================================
def collect_eval_data(model, val_loader, cfg):
    """从模型和 val_loader 收集所有评估所需的数据."""
    from mVAE_aligned import measure_x_conditionality
    from mVAE_common import compute_posterior_accuracy

    model.eval()
    device = cfg.device
    K = cfg.num_classes

    all_mu, all_labels, all_preds = [], [], []
    all_recon_pairs = []  # (original, reconstructed)

    with torch.no_grad():
        for y_img, y_label in val_loader:
            y_img = y_img.to(device)
            mu, logvar = model.enc(y_img)
            z = mu  # 用 mu 做确定性推断

            # 后验
            from mVAE_common import compute_recon_loglik
            recon_loglik, recon_images = compute_recon_loglik(
                model.dec, z, y_img, K, cfg.use_bce)
            log_pi = torch.log(model.pi + 1e-9)
            logits = log_pi.unsqueeze(0) + recon_loglik
            pred = logits.argmax(dim=1)

            all_mu.append(mu.cpu().numpy())
            all_labels.append(y_label.numpy())
            all_preds.append(pred.cpu().numpy())

            # 收集重构样本 (只取前几个 batch)
            if len(all_recon_pairs) < 3:
                best_recon = recon_images[
                    torch.arange(y_img.size(0)), pred].cpu()
                all_recon_pairs.append((y_img.cpu(), best_recon))

    all_mu = np.concatenate(all_mu)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Hungarian 对齐
    post_acc, mapping = compute_posterior_accuracy(all_preds, all_labels, K)

    # x-conditionality
    xcond = measure_x_conditionality(model, val_loader, cfg)

    return {
        "mu": all_mu,
        "labels": all_labels,
        "preds": all_preds,
        "mapping": mapping,  # cluster_id → true_label
        "post_acc": post_acc,
        "xcond": xcond,
        "recon_pairs": all_recon_pairs,
        "pi": model.pi.detach().cpu().numpy(),
    }


# ============================================================
# fig01 — Training Curves (4 panels)
# ============================================================
def plot_training_curves(logger, save_path, **kw):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("mVAE — Training Curves", fontsize=16, fontweight='bold', y=0.98)

    records = logger.records
    epochs = records['epoch']

    # (a) Total + Recon + KL
    ax = axes[0, 0]
    ax.plot(epochs, records['loss'], color=STYLE["color_primary"],
            lw=STYLE["linewidth"], label='Val Loss')
    ax.plot(epochs, records['recon_loss'], color=STYLE["color_accent"],
            lw=1.2, ls='--', label='Recon')
    ax.plot(epochs, records['kl_loss'], color='#7E57C2',
            lw=1.2, ls='-.', label='KL')
    bal = records.get('balance_loss', [])
    if bal and any(v and v != 0 for v in bal):
        ax.plot(epochs, [v if v else 0 for v in bal],
                color=STYLE["color_secondary"], lw=1.2, label='Balance')
    ax.set_title("(a) Training Losses"); ax.legend(fontsize=8)

    # (b) NMI + Posterior Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, records['nmi'], color=STYLE["color_primary"],
            lw=STYLE["linewidth"], label='NMI')
    ax.plot(epochs, records['posterior_acc'], color=STYLE["color_accent"],
            lw=STYLE["linewidth"], ls='--', label='Post. Acc')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) Clustering Quality"); ax.legend()

    # (c) Pi Entropy
    ax = axes[1, 0]
    ax.plot(epochs, records.get('pi_entropy', [0]*len(epochs)),
            color=STYLE["color_tertiary"], lw=STYLE["linewidth"])
    K = 10  # default
    if records.get('pi_values') and len(records['pi_values']) > 0:
        first_pi = records['pi_values'][0]
        if first_pi is not None:
            K = len(first_pi) if isinstance(first_pi, (list, np.ndarray)) else 10
    ax.axhline(y=np.log(K), color='black', ls='--', alpha=0.4,
               label=f'Uniform ($\\ln {K}$)')
    ax.set_title("(c) $\\Pi$ Entropy"); ax.legend()

    # (d) Gumbel τ + Resp Entropy
    ax = axes[1, 1]
    ax.plot(epochs, records.get('tau', []), color='#845ef7',
            lw=STYLE["linewidth"], label='$\\tau$')
    ax2 = ax.twinx()
    ax2.plot(epochs, records.get('resp_entropy', []),
             color=STYLE["color_secondary"], lw=1.2, alpha=0.7, label='Resp Entropy')
    ax.set_title("(d) Gumbel $\\tau$ & Resp Entropy")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig02 — Confusion Matrix
# ============================================================
def plot_confusion_matrix(preds, labels, mapping, save_path, K=10, **kw):
    # mapping: cluster_id → true_label
    inv_map = {v: k for k, v in mapping.items()}  # true_label → cluster_id
    # 对齐: 把 pred 映射到 true label 空间
    aligned_pred = np.array([mapping.get(p, 0) for p in preds])

    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(labels, aligned_pred):
        if 0 <= t < K and 0 <= p < K:
            conf[int(t), int(p)] += 1

    row_sums = conf.sum(axis=1, keepdims=True)
    conf_norm = conf / np.maximum(row_sums, 1)
    acc = np.trace(conf) / max(conf.sum(), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf, cmap='Blues', aspect='equal')
    ax.set_title(f"mVAE — Confusion Matrix (Acc={acc:.1%})",
                 fontsize=STYLE["font_title"], fontweight='bold')
    ax.set_xlabel("Predicted (Aligned) Digit")
    ax.set_ylabel("True Digit Label")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))

    for i in range(K):
        for j in range(K):
            color = 'white' if conf[i, j] > conf.max() * 0.5 else 'black'
            ax.text(j, i, f'{conf[i,j]}', ha='center', va='center',
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig03 — Conditional Generation Grid
# ============================================================
def plot_conditional_generation(model, cfg, mapping, save_path,
                                n_per_class=10, **kw):
    model.eval()
    device = cfg.device
    K = cfg.num_classes
    K_display = min(K, 10)

    z = torch.randn(n_per_class, cfg.latent_dim).to(device)

    # mapping: cluster → digit, 需要反转: digit → cluster
    digit_to_cluster = {}
    for cluster_id, digit in mapping.items():
        if digit not in digit_to_cluster:
            digit_to_cluster[digit] = cluster_id

    fig, axes = plt.subplots(K_display, n_per_class,
                             figsize=(n_per_class * 1.2, K_display * 1.2))

    with torch.no_grad():
        for d in range(K_display):
            ck = digit_to_cluster.get(d, d)
            y_oh = F.one_hot(torch.full((n_per_class,), ck,
                             device=device, dtype=torch.long), K).float()
            imgs = model.dec(z, y_oh).cpu()

            for j in range(n_per_class):
                ax = axes[d, j] if K_display > 1 else axes[j]
                ax.imshow(imgs[j, 0], cmap='gray')
                ax.axis('off')
            if K_display > 1:
                axes[d, 0].set_ylabel(f'{d}', fontsize=10, rotation=0,
                                       labelpad=15, fontweight='bold')

    fig.suptitle("Conditional Generation (Rows: digit $x$, Cols: random $z$)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig04 — Latent Space
# ============================================================
def plot_latent_space(mu, labels, save_path, K=10, **kw):
    fig, ax = plt.subplots(figsize=(9, 8))

    if mu.shape[1] == 2:
        z_2d = mu
        xlabel, ylabel = "$z_1$", "$z_2$"
        title = "mVAE — Latent Space ($d_z=2$)"
    else:
        from sklearn.manifold import TSNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = tsne.fit_transform(mu[:5000])  # 限制样本数
        labels = labels[:5000]
        xlabel, ylabel = "t-SNE 1", "t-SNE 2"
        title = f"mVAE — t-SNE of Latent Space ($d_z={mu.shape[1]}$)"

    for d in range(K):
        mask = labels == d
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=[_tab10(d, K)], s=5, alpha=0.4, label=f'{d}')

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=STYLE["font_title"], fontweight='bold')
    ax.legend(markerscale=4, fontsize=STYLE["font_legend"], ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig05 — Class Frequency (π distribution)
# ============================================================
def plot_class_frequency(pi, mapping, save_path, **kw):
    K = len(pi)
    K_display = min(K, 20)

    # 按 digit 排列
    digit_to_cluster = {}
    for c, d in mapping.items():
        if d not in digit_to_cluster:
            digit_to_cluster[d] = c

    fig, ax = plt.subplots(figsize=(10, 5))

    if K <= 10:
        digits = list(range(K))
        freq = [pi[digit_to_cluster.get(d, d)] for d in digits]
        colors = [_tab10(d, K) for d in digits]
        xlabels = [f'D{d}\n(S{digit_to_cluster.get(d, d)})' for d in digits]
    else:
        # K > 10: 直接按 state 排, 标注 mapping
        digits = list(range(K_display))
        freq = [pi[k] for k in digits]
        colors = [_tab10(mapping.get(k, k) % 10, 10) for k in digits]
        xlabels = [f'S{k}→D{mapping.get(k, "?")}' for k in digits]

    bars = ax.bar(digits, freq, color=colors, edgecolor='white', lw=0.8)
    ax.axhline(y=1.0/K, color='black', ls='--', alpha=0.5,
               label=f'Uniform ({1/K:.1%})')

    for bar, val in zip(bars, freq):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.1%}', ha='center', fontsize=8)

    ax.set_xlabel("Class"); ax.set_ylabel("$\\pi_k$")
    ax.set_title("mVAE — Learned Class Prior $\\Pi$",
                 fontsize=STYLE["font_title"], fontweight='bold')
    ax.set_xticks(digits); ax.set_xticklabels(xlabels, fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig06 — Per-Class Reconstruction
# ============================================================
def plot_per_class_recon(model, val_loader, cfg, mapping, save_path,
                         n_per_class=5, **kw):
    model.eval()
    K = cfg.num_classes
    device = cfg.device
    K_display = min(K, 10)

    digit_to_cluster = {}
    for c, d in mapping.items():
        if d not in digit_to_cluster:
            digit_to_cluster[d] = c

    class_imgs = {k: [] for k in range(K_display)}
    with torch.no_grad():
        for y_img, y_label in val_loader:
            for d in range(K_display):
                mask = y_label == d
                if mask.sum() > 0 and len(class_imgs[d]) < n_per_class:
                    class_imgs[d].extend(
                        y_img[mask][:n_per_class - len(class_imgs[d])])
            if all(len(v) >= n_per_class for v in class_imgs.values()):
                break

    fig, axes = plt.subplots(K_display, n_per_class * 2,
                             figsize=(n_per_class * 3, K_display * 1.3))

    with torch.no_grad():
        for d in range(K_display):
            imgs = torch.stack(class_imgs[d][:n_per_class]).to(device)
            mu, _ = model.enc(imgs)
            ck = digit_to_cluster.get(d, d)
            y_oh = F.one_hot(torch.full((len(imgs),), ck,
                             device=device, dtype=torch.long), K).float()
            recon = model.dec(mu, y_oh)

            for i in range(n_per_class):
                axes[d, i*2].imshow(imgs[i, 0].cpu(), cmap='gray')
                axes[d, i*2].axis('off')
                axes[d, i*2+1].imshow(recon[i, 0].cpu(), cmap='gray')
                axes[d, i*2+1].axis('off')
            axes[d, 0].set_ylabel(f'{d}', fontsize=10, rotation=0, labelpad=15)

    fig.suptitle("Per-Class Reconstruction (Orig → Recon)", fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig07 — x-Conditionality Grid
# ============================================================
def plot_x_conditionality_grid(model, val_loader, cfg, save_path,
                                n_z=5, **kw):
    model.eval()
    K = cfg.num_classes
    K_display = min(K, 10)
    device = cfg.device

    with torch.no_grad():
        y_img, _ = next(iter(val_loader))
        mu, _ = model.enc(y_img[:n_z].to(device))

    fig, axes = plt.subplots(K_display, n_z,
                             figsize=(n_z * 1.8, K_display * 1.5))

    with torch.no_grad():
        for j in range(n_z):
            z = mu[j:j+1]
            for k in range(K_display):
                y_oh = F.one_hot(torch.tensor([k], device=device), K).float()
                img = model.dec(z, y_oh)
                axes[k, j].imshow(img[0, 0].cpu(), cmap='gray')
                axes[k, j].axis('off')
            axes[0, j].set_title(f'$z_{j}$', fontsize=9)
        for k in range(K_display):
            axes[k, 0].set_ylabel(f'$x={k}$', fontsize=10,
                                   rotation=0, labelpad=25)

    from mVAE_aligned import measure_x_conditionality
    xcond = measure_x_conditionality(model, val_loader, cfg)
    fig.suptitle(f"x-Conditionality (score={xcond:.3f}): "
                 f"same $z$, varying $x$",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig08 — Pi Evolution
# ============================================================
def plot_pi_evolution(logger, save_path, **kw):
    pi_values = logger.records.get('pi_values', [])
    epochs = logger.records.get('epoch', [])

    valid = [(ep, pv) for ep, pv in zip(epochs, pi_values)
             if pv is not None and isinstance(pv, (list, np.ndarray))]
    if not valid:
        print("  [fig08] No pi evolution data — skipped")
        return

    eps = [v[0] for v in valid]
    pis = np.array([v[1] for v in valid])
    K = pis.shape[1]

    fig, ax = plt.subplots(figsize=(12, 6))
    for k in range(K):
        ax.plot(eps, pis[:, k], color=_tab10(k, K), lw=1.5, alpha=0.8,
                label=f'$\\pi_{{{k}}}$')

    ax.axhline(y=1.0/K, color='black', ls='--', alpha=0.4,
               label=f'Uniform ({1/K:.2f})')
    ax.set_xlabel("Epoch"); ax.set_ylabel("$\\pi_k$")
    ax.set_title("mVAE — Class Prior $\\Pi$ Evolution",
                 fontsize=STYLE["font_title"], fontweight='bold')
    ax.legend(fontsize=7, ncol=K//2 + 1, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig09 — Resp Entropy & τ
# ============================================================
def plot_resp_entropy_tau(logger, save_path, **kw):
    epochs = logger.records['epoch']
    resp_ent = logger.records.get('resp_entropy', [])
    tau = logger.records.get('tau', [])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    if tau:
        ax1.plot(epochs, tau, color='#845ef7', lw=2, label='$\\tau$')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("$\\tau$", color='#845ef7')
    ax1.tick_params(axis='y', labelcolor='#845ef7')

    if resp_ent:
        ax2 = ax1.twinx()
        ax2.plot(epochs, resp_ent, color=STYLE["color_secondary"],
                 lw=1.5, alpha=0.7, label='Resp Entropy')
        ax2.set_ylabel("Resp Entropy", color=STYLE["color_secondary"])
        ax2.tick_params(axis='y', labelcolor=STYLE["color_secondary"])
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2)
    else:
        ax1.legend()

    ax1.set_title("mVAE — Gumbel $\\tau$ & Posterior Sharpness",
                  fontsize=STYLE["font_title"], fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig10 — Recon Samples
# ============================================================
def plot_recon_samples(recon_pairs, save_path, n_show=10, **kw):
    if not recon_pairs:
        print("  [fig10] No recon data — skipped")
        return

    orig_batch, recon_batch = recon_pairs[0]
    n = min(n_show, orig_batch.size(0))

    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 3.5))
    fig.suptitle("mVAE — Reconstruction Quality",
                 fontsize=STYLE["font_title"], fontweight='bold', y=1.04)

    for i in range(n):
        axes[0, i].imshow(orig_batch[i, 0], cmap='gray'); axes[0, i].axis('off')
        axes[1, i].imshow(recon_batch[i, 0], cmap='gray'); axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Input", fontsize=10)
    axes[1, 0].set_ylabel("Recon", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig11 — Per-Digit Accuracy
# ============================================================
def plot_per_digit_accuracy(preds, labels, mapping, save_path, K=10, **kw):
    aligned_pred = np.array([mapping.get(p, 0) for p in preds])

    fig, ax = plt.subplots(figsize=(10, 5))
    accs = []
    for d in range(K):
        mask = labels == d
        if mask.sum() > 0:
            acc = np.mean(aligned_pred[mask] == d)
        else:
            acc = 0
        accs.append(acc)

    colors = [_tab10(d, K) for d in range(K)]
    bars = ax.bar(range(K), accs, color=colors, edgecolor='white', lw=0.8)
    ax.axhline(y=np.mean(accs), color='black', ls='--', alpha=0.5,
               label=f'Mean ({np.mean(accs):.1%})')

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', fontsize=9)

    ax.set_xlabel("Digit"); ax.set_ylabel("Accuracy")
    ax.set_title("mVAE — Per-Digit Posterior Accuracy",
                 fontsize=STYLE["font_title"], fontweight='bold')
    ax.set_xticks(range(K)); ax.set_ylim(0, 1.1); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig12 — Balance Loss Effect (Pi Gini coefficient over epochs)
# ============================================================
def plot_balance_effect(logger, save_path, **kw):
    epochs = logger.records.get('epoch', [])
    pi_values = logger.records.get('pi_values', [])
    bal_loss = logger.records.get('balance_loss', [])

    valid = [(ep, pv, bl) for ep, pv, bl in zip(epochs, pi_values, bal_loss)
             if pv is not None and isinstance(pv, (list, np.ndarray))]
    if not valid:
        print("  [fig12] No balance data — skipped")
        return

    eps = [v[0] for v in valid]
    pis = np.array([v[1] for v in valid])
    bals = [v[2] if v[2] else 0 for v in valid]

    # Gini coefficient of pi as uniformity measure
    def gini(pi):
        pi_sorted = np.sort(pi)
        n = len(pi_sorted)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * pi_sorted) / (n * np.sum(pi_sorted))) - (n + 1) / n

    ginis = [gini(p) for p in pis]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(eps, ginis, color=STYLE["color_primary"], lw=2,
             label='$\\Pi$ Gini (0=uniform)')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gini Coefficient", color=STYLE["color_primary"])
    ax1.set_ylim(0, max(ginis) * 1.3 if max(ginis) > 0 else 1)

    if any(b != 0 for b in bals):
        ax2 = ax1.twinx()
        ax2.plot(eps, bals, color=STYLE["color_tertiary"], lw=1.5,
                 alpha=0.6, label='Balance Loss')
        ax2.set_ylabel("Balance Loss", color=STYLE["color_tertiary"])
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2)
    else:
        ax1.legend()

    ax1.set_title("mVAE — Balance Loss & Class Uniformity",
                  fontsize=STYLE["font_title"], fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# generate_all_mVAE_figures — 统一入口
# ============================================================
def generate_all_mVAE_figures(model, logger, val_loader, cfg,
                               save_dir="figures", mode="unsup"):
    """
    一键生成全部 12 张论文图表.

    参数:
      model: 训练好的 mVAE 模型
      logger: TrainingLogger (来自 mVAE_common)
      val_loader: 验证集 DataLoader
      cfg: Config 对象
      save_dir: 输出目录
      mode: "unsup" 或 "semisup"
    """
    os.makedirs(save_dir, exist_ok=True)
    K = cfg.num_classes

    print(f"\n{'='*50}")
    print(f"Generating all mVAE figures ({mode})")
    print(f"{'='*50}")

    # 收集评估数据
    print("Collecting evaluation data...")
    eval_data = collect_eval_data(model, val_loader, cfg)

    mapping = eval_data["mapping"]
    print(f"  Post. Accuracy: {eval_data['post_acc']:.4f}")
    print(f"  x-Conditionality: {eval_data['xcond']:.4f}")
    print(f"  Mapping (cluster→digit): {mapping}")

    # fig01
    plot_training_curves(
        logger, os.path.join(save_dir, "fig01_training_curves.png"))

    # fig02
    plot_confusion_matrix(
        eval_data["preds"], eval_data["labels"], mapping,
        os.path.join(save_dir, "fig02_confusion_matrix.png"), K=K)

    # fig03
    plot_conditional_generation(
        model, cfg, mapping,
        os.path.join(save_dir, "fig03_conditional_generation.png"))

    # fig04
    plot_latent_space(
        eval_data["mu"], eval_data["labels"],
        os.path.join(save_dir, "fig04_latent_space.png"), K=K)

    # fig05
    plot_class_frequency(
        eval_data["pi"], mapping,
        os.path.join(save_dir, "fig05_class_frequency.png"))

    # fig06
    plot_per_class_recon(
        model, val_loader, cfg, mapping,
        os.path.join(save_dir, "fig06_per_class_recon.png"))

    # fig07
    plot_x_conditionality_grid(
        model, val_loader, cfg,
        os.path.join(save_dir, "fig07_x_conditionality.png"))

    # fig08
    plot_pi_evolution(
        logger, os.path.join(save_dir, "fig08_pi_evolution.png"))

    # fig09
    plot_resp_entropy_tau(
        logger, os.path.join(save_dir, "fig09_resp_entropy_tau.png"))

    # fig10
    plot_recon_samples(
        eval_data["recon_pairs"],
        os.path.join(save_dir, "fig10_recon_samples.png"))

    # fig11
    plot_per_digit_accuracy(
        eval_data["preds"], eval_data["labels"], mapping,
        os.path.join(save_dir, "fig11_per_digit_accuracy.png"), K=K)

    # fig12
    plot_balance_effect(
        logger, os.path.join(save_dir, "fig12_balance_effect.png"))

    print(f"\n>> All {12} figures saved to {save_dir}/")
