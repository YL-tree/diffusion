"""
mVAE_plot_utils.py — mVAE 论文级可视化 (unsup + semisup 通用)
================================================================

图表清单 (共 12 张):
  fig01  Training Curves        — Loss 分量 / NMI+Acc / Pi Entropy / τ
  fig02  Confusion Matrix       — 后验预测 vs 真实标签
  fig03  Conditional Gen Grid   — 行=x, 列=不同 z 采样
  fig04  Latent Space           — t-SNE 或 2D scatter
  fig05  Class Frequency (π)    — 学到的先验分布
  fig06  Per-Class Recon        — 原图 → 重构
  fig07  x-Conditionality Grid  — 固定 z 变 x
  fig08  Pi Evolution           — π 随 epoch 变化
  fig09  Resp Entropy & τ       — 后验 sharpness 与温度退火
  fig10  Recon Samples          — 随机输入 vs 重构
  fig11  Per-Digit Accuracy     — 逐数字准确率
  fig12  Balance Loss Effect    — balance loss 与 π 均衡性

用法:
  from mVAE_plot_utils import generate_all_mVAE_figures
"""

import os
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
STYLE = dict(
    fig_dpi=200, font_title=15, font_label=12, font_tick=10, font_legend=9,
    color_primary="#2196F3", color_secondary="#4CAF50",
    color_tertiary="#FF9800", color_accent="#E91E63",
    linewidth=1.8,
)
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

def _tab10(k, K=10):
    return plt.cm.tab10(k / K)


# ============================================================
# 数据收集 (unsup / semisup 通用)
# ============================================================
def collect_eval_data(model, val_loader, cfg):
    """收集所有评估所需数据, 与训练模式无关 (总是用 forward_unlabeled 做推断)."""
    from mVAE_aligned import measure_x_conditionality
    from mVAE_common import compute_posterior_accuracy, compute_recon_loglik

    model.eval()
    device = cfg.device
    K = cfg.num_classes

    all_mu, all_labels, all_preds, recon_pairs = [], [], [], []

    with torch.no_grad():
        for y_img, y_label in val_loader:
            y_img = y_img.to(device)
            mu, logvar = model.enc(y_img)
            z = mu

            recon_loglik, recon_images = compute_recon_loglik(
                model.dec, z, y_img, K, cfg.use_bce)
            log_pi = torch.log(model.pi + 1e-9)
            logits = log_pi.unsqueeze(0) + recon_loglik
            pred = logits.argmax(dim=1)

            all_mu.append(mu.cpu().numpy())
            all_labels.append(y_label.numpy())
            all_preds.append(pred.cpu().numpy())

            if len(recon_pairs) < 3:
                best_recon = recon_images[
                    torch.arange(y_img.size(0)), pred].cpu()
                recon_pairs.append((y_img.cpu(), best_recon))

    all_mu    = np.concatenate(all_mu)
    all_labels= np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    post_acc, mapping = compute_posterior_accuracy(all_preds, all_labels, K)
    xcond = measure_x_conditionality(model, val_loader, cfg)

    return dict(
        mu=all_mu, labels=all_labels, preds=all_preds,
        mapping=mapping, post_acc=post_acc, xcond=xcond,
        recon_pairs=recon_pairs,
        pi=model.pi.detach().cpu().numpy(),
    )


# ============================================================
# fig01 — Training Curves (4 panels)
# ============================================================
def plot_training_curves(logger, save_path, mode="unsup", **kw):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"mVAE Training Curves ({mode})"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    rec = logger.records
    epochs = rec['epoch']

    ax = axes[0, 0]
    ax.plot(epochs, rec['loss'], color=STYLE["color_primary"], lw=1.8, label='Val Loss')
    ax.plot(epochs, rec['recon_loss'], color=STYLE["color_accent"],
            lw=1.2, ls='--', label='Recon')
    ax.plot(epochs, rec['kl_loss'], color='#7E57C2', lw=1.2, ls='-.', label='KL')
    bal = rec.get('balance_loss', [])
    if bal and any(v and v != 0 for v in bal):
        ax.plot(epochs, [v or 0 for v in bal],
                color=STYLE["color_secondary"], lw=1.2, label='Balance')
    ax.set_title("(a) Losses"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(epochs, rec['nmi'], color=STYLE["color_primary"], lw=1.8, label='NMI')
    ax.plot(epochs, rec['posterior_acc'], color=STYLE["color_accent"],
            lw=1.8, ls='--', label='Post. Acc')
    ax.set_ylim(-0.05, 1.05); ax.set_title("(b) Clustering Quality"); ax.legend()

    ax = axes[1, 0]
    pi_ent = rec.get('pi_entropy', [0]*len(epochs))
    ax.plot(epochs, pi_ent, color=STYLE["color_tertiary"], lw=1.8)
    K = 10
    pv = rec.get('pi_values', [])
    if pv and pv[0] is not None:
        K = len(pv[0]) if isinstance(pv[0], (list, np.ndarray)) else 10
    ax.axhline(y=np.log(K), color='black', ls='--', alpha=0.4,
               label=f'Uniform ($\\ln {K}$)')
    ax.set_title("(c) $\\Pi$ Entropy"); ax.legend()

    ax = axes[1, 1]
    tau = rec.get('tau', [])
    if tau: ax.plot(epochs, tau, color='#845ef7', lw=1.8, label='$\\tau$')
    ax2 = ax.twinx()
    re = rec.get('resp_entropy', [])
    if re:
        ax2.plot(epochs, re, color=STYLE["color_secondary"],
                 lw=1.2, alpha=0.7, label='Resp Ent')
    ax.set_title("(d) Gumbel $\\tau$ & Resp Entropy")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig02 — Confusion Matrix
# ============================================================
def plot_confusion_matrix(preds, labels, mapping, save_path, K=10, mode="unsup", **kw):
    aligned_pred = np.array([mapping.get(p, 0) for p in preds])
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(labels, aligned_pred):
        if 0 <= t < K and 0 <= p < K:
            conf[int(t), int(p)] += 1

    acc = np.trace(conf) / max(conf.sum(), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf, cmap='Blues', aspect='equal')
    ax.set_title(f"Confusion Matrix ({mode}, Acc={acc:.1%})",
                 fontsize=STYLE["font_title"], fontweight='bold')
    ax.set_xlabel("Predicted (aligned)"); ax.set_ylabel("True Label")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    for i in range(K):
        for j in range(K):
            c = 'white' if conf[i,j] > conf.max()*0.5 else 'black'
            ax.text(j, i, str(conf[i,j]), ha='center', va='center', fontsize=8, color=c)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig03 — Conditional Generation
# ============================================================
def plot_conditional_generation(model, cfg, mapping, save_path, n_per_class=10, **kw):
    model.eval(); device = cfg.device; K = cfg.num_classes
    K_disp = min(K, 10)
    z = torch.randn(n_per_class, cfg.latent_dim).to(device)

    d2c = {}
    for c, d in mapping.items():
        if d not in d2c: d2c[d] = c

    fig, axes = plt.subplots(K_disp, n_per_class,
                             figsize=(n_per_class*1.2, K_disp*1.2))
    with torch.no_grad():
        for d in range(K_disp):
            ck = d2c.get(d, d)
            yoh = F.one_hot(torch.full((n_per_class,), ck,
                            device=device, dtype=torch.long), K).float()
            imgs = model.dec(z, yoh).cpu()
            for j in range(n_per_class):
                ax = axes[d, j] if K_disp > 1 else axes[j]
                ax.imshow(imgs[j, 0], cmap='gray'); ax.axis('off')
            if K_disp > 1:
                axes[d, 0].set_ylabel(f'{d}', fontsize=10, rotation=0, labelpad=15)
    fig.suptitle("Conditional Gen (row=$x$, col=random $z$)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig04 — Latent Space
# ============================================================
def plot_latent_space(mu, labels, save_path, K=10, **kw):
    fig, ax = plt.subplots(figsize=(9, 8))
    if mu.shape[1] == 2:
        z2d, xl, yl = mu, "$z_1$", "$z_2$"
        title = "Latent Space ($d_z=2$)"
    else:
        from sklearn.manifold import TSNE
        print("  Computing t-SNE...")
        z2d = TSNE(n_components=2, random_state=42).fit_transform(mu[:5000])
        labels = labels[:5000]
        xl, yl = "t-SNE 1", "t-SNE 2"
        title = f"t-SNE ($d_z={mu.shape[1]}$)"
    for d in range(K):
        m = labels == d
        if m.sum() > 0:
            ax.scatter(z2d[m, 0], z2d[m, 1], c=[_tab10(d, K)], s=5, alpha=0.4, label=f'{d}')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"mVAE — {title}", fontsize=STYLE["font_title"], fontweight='bold')
    ax.legend(markerscale=4, fontsize=STYLE["font_legend"], ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig05 — Class Frequency
# ============================================================
def plot_class_frequency(pi, mapping, save_path, **kw):
    K = len(pi)
    d2c = {}
    for c, d in mapping.items():
        if d not in d2c: d2c[d] = c

    fig, ax = plt.subplots(figsize=(10, 5))
    if K <= 10:
        digits = list(range(K))
        freq = [pi[d2c.get(d, d)] for d in digits]
        colors = [_tab10(d, K) for d in digits]
        xlbl = [f'D{d}\n(S{d2c.get(d,d)})' for d in digits]
    else:
        digits = list(range(K))
        freq = [pi[k] for k in digits]
        colors = [_tab10(mapping.get(k, k) % 10, 10) for k in digits]
        xlbl = [f'S{k}' for k in digits]

    bars = ax.bar(digits, freq, color=colors, edgecolor='white', lw=0.8)
    ax.axhline(y=1/K, color='black', ls='--', alpha=0.5, label=f'Uniform ({1/K:.1%})')
    for b, v in zip(bars, freq):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                f'{v:.1%}', ha='center', fontsize=8)
    ax.set_xlabel("Class"); ax.set_ylabel("$\\pi_k$")
    ax.set_title("Learned Class Prior $\\Pi$", fontsize=STYLE["font_title"], fontweight='bold')
    ax.set_xticks(digits); ax.set_xticklabels(xlbl, fontsize=8); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig06 — Per-Class Recon
# ============================================================
def plot_per_class_recon(model, val_loader, cfg, mapping, save_path, n=5, **kw):
    model.eval(); K = cfg.num_classes; device = cfg.device
    K_disp = min(K, 10)
    d2c = {}
    for c, d in mapping.items():
        if d not in d2c: d2c[d] = c

    imgs = {k: [] for k in range(K_disp)}
    with torch.no_grad():
        for y, lab in val_loader:
            for d in range(K_disp):
                m = lab == d
                if m.sum() > 0 and len(imgs[d]) < n:
                    imgs[d].extend(y[m][:n - len(imgs[d])])
            if all(len(v) >= n for v in imgs.values()): break

    fig, axes = plt.subplots(K_disp, n*2, figsize=(n*3, K_disp*1.3))
    with torch.no_grad():
        for d in range(K_disp):
            batch = torch.stack(imgs[d][:n]).to(device)
            mu, _ = model.enc(batch)
            ck = d2c.get(d, d)
            yoh = F.one_hot(torch.full((len(batch),), ck,
                            device=device, dtype=torch.long), K).float()
            rec = model.dec(mu, yoh)
            for i in range(n):
                axes[d, i*2].imshow(batch[i,0].cpu(), cmap='gray'); axes[d, i*2].axis('off')
                axes[d, i*2+1].imshow(rec[i,0].cpu(), cmap='gray'); axes[d, i*2+1].axis('off')
            axes[d, 0].set_ylabel(f'{d}', fontsize=10, rotation=0, labelpad=15)
    fig.suptitle("Per-Class Recon (Orig→Recon)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig07 — x-Conditionality Grid
# ============================================================
def plot_x_conditionality_grid(model, val_loader, cfg, save_path, n_z=5, **kw):
    model.eval(); K = cfg.num_classes; device = cfg.device
    K_disp = min(K, 10)
    with torch.no_grad():
        y, _ = next(iter(val_loader))
        mu, _ = model.enc(y[:n_z].to(device))

    fig, axes = plt.subplots(K_disp, n_z, figsize=(n_z*1.8, K_disp*1.5))
    with torch.no_grad():
        for j in range(n_z):
            z = mu[j:j+1]
            for k in range(K_disp):
                yoh = F.one_hot(torch.tensor([k], device=device), K).float()
                img = model.dec(z, yoh)
                axes[k, j].imshow(img[0,0].cpu(), cmap='gray'); axes[k, j].axis('off')
            axes[0, j].set_title(f'$z_{j}$', fontsize=9)
        for k in range(K_disp):
            axes[k, 0].set_ylabel(f'$x={k}$', fontsize=10, rotation=0, labelpad=25)

    from mVAE_aligned import measure_x_conditionality
    xc = measure_x_conditionality(model, val_loader, cfg)
    fig.suptitle(f"x-Conditionality (score={xc:.3f}): same $z$, varying $x$",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# fig08–12: 省略重复模板, 与之前版本相同 (pi evolution, resp entropy,
#           recon samples, per-digit accuracy, balance effect)
# ============================================================
def plot_pi_evolution(logger, save_path, **kw):
    pv = logger.records.get('pi_values', [])
    ep = logger.records.get('epoch', [])
    valid = [(e, p) for e, p in zip(ep, pv) if p is not None and isinstance(p, (list, np.ndarray))]
    if not valid: print("  [fig08] skip"); return
    eps = [v[0] for v in valid]; pis = np.array([v[1] for v in valid]); K = pis.shape[1]
    fig, ax = plt.subplots(figsize=(12, 6))
    for k in range(K):
        ax.plot(eps, pis[:, k], color=_tab10(k, K), lw=1.5, alpha=0.8, label=f'$\\pi_{{{k}}}$')
    ax.axhline(y=1/K, color='black', ls='--', alpha=0.4, label=f'Uniform')
    ax.set_xlabel("Epoch"); ax.set_ylabel("$\\pi_k$")
    ax.set_title("$\\Pi$ Evolution", fontsize=STYLE["font_title"], fontweight='bold')
    ax.legend(fontsize=7, ncol=K//2+1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


def plot_recon_samples(recon_pairs, save_path, n=10, **kw):
    if not recon_pairs: print("  [fig10] skip"); return
    orig, rec = recon_pairs[0]; n = min(n, orig.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n*1.8, 3.5))
    fig.suptitle("Reconstruction", fontsize=STYLE["font_title"], fontweight='bold', y=1.04)
    for i in range(n):
        axes[0,i].imshow(orig[i,0], cmap='gray'); axes[0,i].axis('off')
        axes[1,i].imshow(rec[i,0], cmap='gray'); axes[1,i].axis('off')
    axes[0,0].set_ylabel("Input", fontsize=10); axes[1,0].set_ylabel("Recon", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


def plot_per_digit_accuracy(preds, labels, mapping, save_path, K=10, **kw):
    aligned = np.array([mapping.get(p, 0) for p in preds])
    fig, ax = plt.subplots(figsize=(10, 5))
    accs = []
    for d in range(K):
        m = labels == d
        accs.append(np.mean(aligned[m] == d) if m.sum() > 0 else 0)
    bars = ax.bar(range(K), accs, color=[_tab10(d, K) for d in range(K)], edgecolor='white')
    ax.axhline(y=np.mean(accs), color='black', ls='--', alpha=0.5, label=f'Mean ({np.mean(accs):.1%})')
    for b, v in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.1%}', ha='center', fontsize=9)
    ax.set_xlabel("Digit"); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Digit Accuracy", fontsize=STYLE["font_title"], fontweight='bold')
    ax.set_xticks(range(K)); ax.set_ylim(0, 1.1); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


def plot_resp_entropy_tau(logger, save_path, **kw):
    ep = logger.records['epoch']; tau = logger.records.get('tau', [])
    re = logger.records.get('resp_entropy', [])
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if tau: ax1.plot(ep, tau, color='#845ef7', lw=2, label='$\\tau$')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("$\\tau$", color='#845ef7')
    if re:
        ax2 = ax1.twinx()
        ax2.plot(ep, re, color=STYLE["color_secondary"], lw=1.5, alpha=0.7, label='Resp Ent')
        ax2.set_ylabel("Resp Entropy"); h2, l2 = ax2.get_legend_handles_labels()
    else: h2, l2 = [], []
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    ax1.set_title("$\\tau$ & Posterior Sharpness", fontsize=STYLE["font_title"], fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


def plot_balance_effect(logger, save_path, **kw):
    ep = logger.records.get('epoch', []); pv = logger.records.get('pi_values', [])
    bl = logger.records.get('balance_loss', [])
    valid = [(e, p, b) for e, p, b in zip(ep, pv, bl)
             if p is not None and isinstance(p, (list, np.ndarray))]
    if not valid: print("  [fig12] skip"); return
    eps = [v[0] for v in valid]; pis = np.array([v[1] for v in valid])
    bals = [v[2] or 0 for v in valid]
    def gini(pi):
        s = np.sort(pi); n = len(s); idx = np.arange(1, n+1)
        return (2*np.sum(idx*s)/(n*np.sum(s))) - (n+1)/n
    ginis = [gini(p) for p in pis]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(eps, ginis, color=STYLE["color_primary"], lw=2, label='Gini (0=uniform)')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Gini", color=STYLE["color_primary"])
    if any(b != 0 for b in bals):
        ax2 = ax1.twinx()
        ax2.plot(eps, bals, color=STYLE["color_tertiary"], lw=1.5, alpha=0.6, label='Balance Loss')
        ax2.set_ylabel("Balance Loss"); h2, l2 = ax2.get_legend_handles_labels()
    else: h2, l2 = [], []
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    ax1.set_title("Balance Loss & Uniformity", fontsize=STYLE["font_title"], fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=STYLE["fig_dpi"], bbox_inches='tight'); plt.close()
    print(f"  Saved {save_path}")


# ============================================================
# 统一入口
# ============================================================
def generate_all_mVAE_figures(model, logger, val_loader, cfg,
                               save_dir="figures", mode="unsup"):
    """
    一键生成全部 12 张论文图表.

    mode: "unsup" | "semisup" — 影响标题和 phase 标注.
    模型推断方式相同 (forward_unlabeled), 所以 unsup/semisup 通用.
    """
    os.makedirs(save_dir, exist_ok=True)
    K = cfg.num_classes

    print(f"\n{'='*50}")
    print(f"Generating mVAE figures ({mode}, K={K})")
    print(f"{'='*50}")

    print("Collecting eval data...")
    ed = collect_eval_data(model, val_loader, cfg)
    print(f"  Acc={ed['post_acc']:.4f}  xCond={ed['xcond']:.4f}")
    print(f"  Mapping: {ed['mapping']}")

    p = lambda fn: os.path.join(save_dir, fn)

    plot_training_curves(logger, p("fig01_training_curves.png"), mode=mode)
    plot_confusion_matrix(ed["preds"], ed["labels"], ed["mapping"],
                          p("fig02_confusion_matrix.png"), K=K, mode=mode)
    plot_conditional_generation(model, cfg, ed["mapping"],
                                p("fig03_conditional_gen.png"))
    plot_latent_space(ed["mu"], ed["labels"],
                      p("fig04_latent_space.png"), K=K)
    plot_class_frequency(ed["pi"], ed["mapping"],
                         p("fig05_class_frequency.png"))
    plot_per_class_recon(model, val_loader, cfg, ed["mapping"],
                         p("fig06_per_class_recon.png"))
    plot_x_conditionality_grid(model, val_loader, cfg,
                                p("fig07_x_conditionality.png"))
    plot_pi_evolution(logger, p("fig08_pi_evolution.png"))
    plot_resp_entropy_tau(logger, p("fig09_resp_entropy_tau.png"))
    plot_recon_samples(ed["recon_pairs"], p("fig10_recon_samples.png"))
    plot_per_digit_accuracy(ed["preds"], ed["labels"], ed["mapping"],
                            p("fig11_per_digit_accuracy.png"), K=K)
    plot_balance_effect(logger, p("fig12_balance_effect.png"))

    print(f"\n>> All 12 figures saved to {save_dir}/")