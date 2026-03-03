"""
mVAE_ablation.py — mVAE 消融实验自动化框架
=============================================

参考 HMM-VAE 的 ablation_study.py 框架, 为 mVAE (Section 2.2) 定制.

功能:
  1. 定义所有消融配置 (8 个维度, 18 个配置)
  2. 自动运行全部实验 (每个配置 x 3 seeds)
  3. 保存每次运行的指标到 JSON
  4. 生成对比图表和 LaTeX 表格

使用:
  python mVAE_ablation.py                         # 运行全部 + 出图
  python mVAE_ablation.py --plot-only             # 仅从已有结果出图
  python mVAE_ablation.py --config full_model --seed 42
  python mVAE_ablation.py --quick                 # 快速模式 (30 epochs)

依赖: mVAE_aligned.py, mVAE_common.py
"""

import os, sys, json, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mVAE_common import (
    Config, compute_NMI, compute_posterior_accuracy, TrainingLogger
)
from mVAE_aligned import (
    mVAE, get_unsup_loaders, evaluate_model, measure_x_conditionality
)


# ==============================================================
# 消融配置
# ==============================================================
DEFAULT_CONFIG = {
    "name": "full_model",
    "display_name": "Full model",
    "latent_dim": 2,
    "num_classes": 10,
    "beta": 2.0,
    "z_dropout_rate": 0.5,
    "balance_weight": 10.0,
    "decoder_type": "film",
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 100,
    "tau_start": 1.0,
    "tau_min": 0.1,
    "tau_rate": 0.98,
}

ABLATION_CONFIGS = {
    # ═══════════════════════════════════════════
    # 基准
    # ═══════════════════════════════════════════
    "full_model": {**DEFAULT_CONFIG},

    # ═══════════════════════════════════════════
    # 1. latent_dim 消融 — 核心权衡
    #    z 太小 → 重构差; z 太大 → x 退化
    # ═══════════════════════════════════════════
    "zdim_4": {
        **DEFAULT_CONFIG,
        "name": "zdim_4", "display_name": "$d_z=4$",
        "latent_dim": 4,
    },
    "zdim_8": {
        **DEFAULT_CONFIG,
        "name": "zdim_8", "display_name": "$d_z=8$",
        "latent_dim": 8,
    },
    "zdim_16": {
        **DEFAULT_CONFIG,
        "name": "zdim_16", "display_name": "$d_z=16$",
        "latent_dim": 16,
    },

    # ═══════════════════════════════════════════
    # 2. beta 消融 — KL 权重
    # ═══════════════════════════════════════════
    "beta_0.5": {
        **DEFAULT_CONFIG,
        "name": "beta_0.5", "display_name": "$\\beta=0.5$",
        "beta": 0.5,
    },
    "beta_1.0": {
        **DEFAULT_CONFIG,
        "name": "beta_1.0", "display_name": "$\\beta=1.0$",
        "beta": 1.0,
    },
    "beta_4.0": {
        **DEFAULT_CONFIG,
        "name": "beta_4.0", "display_name": "$\\beta=4.0$",
        "beta": 4.0,
    },

    # ═══════════════════════════════════════════
    # 3. 正则化消融 — z-dropout / balance loss
    # ═══════════════════════════════════════════
    "no_z_dropout": {
        **DEFAULT_CONFIG,
        "name": "no_z_dropout", "display_name": "w/o z-dropout",
        "z_dropout_rate": 0.0,
    },
    "no_balance": {
        **DEFAULT_CONFIG,
        "name": "no_balance", "display_name": "w/o balance",
        "balance_weight": 0.0,
    },
    "no_both_reg": {
        **DEFAULT_CONFIG,
        "name": "no_both_reg", "display_name": "w/o both reg.",
        "z_dropout_rate": 0.0, "balance_weight": 0.0,
    },

    # ═══════════════════════════════════════════
    # 4. Decoder 类型
    # ═══════════════════════════════════════════
    "decoder_concat": {
        **DEFAULT_CONFIG,
        "name": "decoder_concat", "display_name": "Concat decoder",
        "decoder_type": "concat",
    },

    # ═══════════════════════════════════════════
    # 5. K 过估计
    # ═══════════════════════════════════════════
    "K_15": {
        **DEFAULT_CONFIG,
        "name": "K_15", "display_name": "$K=15$",
        "num_classes": 15,
    },
    "K_20": {
        **DEFAULT_CONFIG,
        "name": "K_20", "display_name": "$K=20$",
        "num_classes": 20,
    },

    # ═══════════════════════════════════════════
    # 6. Gumbel τ 策略
    # ═══════════════════════════════════════════
    "tau_aggressive": {
        **DEFAULT_CONFIG,
        "name": "tau_aggressive", "display_name": "$\\tau$: aggressive",
        "tau_min": 0.05, "tau_rate": 0.95,
    },
    "tau_conservative": {
        **DEFAULT_CONFIG,
        "name": "tau_conservative", "display_name": "$\\tau$: conservative",
        "tau_min": 0.3, "tau_rate": 0.99,
    },

    # ═══════════════════════════════════════════
    # 7. latent_dim × beta 联合 — 最关键的交互
    # ═══════════════════════════════════════════
    "dim8_beta1": {
        **DEFAULT_CONFIG,
        "name": "dim8_beta1", "display_name": "$d_z=8,\\beta=1$",
        "latent_dim": 8, "beta": 1.0,
    },
    "dim8_beta0.5": {
        **DEFAULT_CONFIG,
        "name": "dim8_beta0.5", "display_name": "$d_z=8,\\beta=0.5$",
        "latent_dim": 8, "beta": 0.5,
    },
    "dim16_beta0.5": {
        **DEFAULT_CONFIG,
        "name": "dim16_beta0.5", "display_name": "$d_z=16,\\beta=0.5$",
        "latent_dim": 16, "beta": 0.5,
    },
}

SEEDS = [42, 123, 2024]


# ==============================================================
# 将 config dict 转换为 Config 对象
# ==============================================================
def _make_cfg(cfg_dict, device='cpu'):
    cfg = Config()
    cfg.latent_dim = cfg_dict["latent_dim"]
    cfg.num_classes = cfg_dict["num_classes"]
    cfg.beta = cfg_dict["beta"]
    cfg.z_dropout_rate = cfg_dict["z_dropout_rate"]
    cfg.balance_weight = cfg_dict["balance_weight"]
    cfg.decoder_type = cfg_dict["decoder_type"]
    cfg.lr = cfg_dict["lr"]
    cfg.batch_size = cfg_dict["batch_size"]
    cfg.final_epochs = cfg_dict["epochs"]
    cfg.init_gumbel_temp = cfg_dict["tau_start"]
    cfg.min_gumbel_temp = cfg_dict["tau_min"]
    cfg.gumbel_anneal_rate = cfg_dict["tau_rate"]
    cfg.current_gumbel_temp = cfg_dict["tau_start"]
    cfg.device = device
    cfg.output_dir = "./tmp_ablation"
    return cfg


# ==============================================================
# 训练单次实验
# ==============================================================
def train_single_run(config, seed, train_loader, val_loader,
                     save_dir="mVAE_ablation_results"):
    """训练一次 mVAE, 返回评估指标 dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_dict = config
    run_name = f"{cfg_dict['name']}_seed{seed}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = _make_cfg(cfg_dict, device)

    print(f"\n{'='*60}")
    print(f"  {cfg_dict['display_name']}  |  seed={seed}")
    print(f"  d_z={cfg.latent_dim}, K={cfg.num_classes}, β={cfg.beta}, "
          f"z_drop={cfg.z_dropout_rate}, bal={cfg.balance_weight}")
    print(f"{'='*60}")

    # ---- 模型 ----
    model = mVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---- 训练 ----
    EPOCHS = cfg_dict["epochs"]
    best_acc = 0.0
    best_model_path = os.path.join(run_dir, "best_model.pt")
    history = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        cfg.current_gumbel_temp = max(
            cfg.min_gumbel_temp,
            cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        ep_metrics = defaultdict(float)
        n_batches = 0

        for y_img, _ in train_loader:
            y_img = y_img.to(device)
            loss, info = model.forward_unlabeled(y_img, cfg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            for k in ['recon', 'kl', 'prior', 'post_corr', 'resp_ent', 'balance']:
                ep_metrics[k] += info.get(k, 0)
            n_batches += 1

        # ---- 评估 ----
        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)

        if post_acc > best_acc:
            best_acc = post_acc
            torch.save(model.state_dict(), best_model_path)

        # x-conditionality (每 20 epoch)
        xcond = 0.0
        if epoch % 20 == 0 or epoch == EPOCHS:
            xcond = measure_x_conditionality(model, val_loader, cfg)

        # pi 熵
        pi_np = model.pi.detach().cpu().numpy()
        pi_entropy = float(-(pi_np * np.log(pi_np + 1e-9)).sum())

        # 活跃类别数 (pi > 0.02)
        n_active = int((pi_np > 0.02).sum())

        history.append({
            "epoch": epoch,
            "loss": val_loss,
            "recon": ep_metrics['recon'] / n_batches,
            "kl": ep_metrics['kl'] / n_batches,
            "prior": ep_metrics['prior'] / n_batches,
            "post_corr": ep_metrics['post_corr'] / n_batches,
            "resp_ent": ep_metrics['resp_ent'] / n_batches,
            "balance": ep_metrics['balance'] / n_batches,
            "nmi": nmi,
            "post_acc": post_acc,
            "xcond": xcond if xcond > 0 else None,
            "pi_entropy": pi_entropy,
            "n_active": n_active,
            "tau": cfg.current_gumbel_temp,
            "pi_values": pi_np.tolist(),
        })

        if epoch % 25 == 0 or epoch == EPOCHS:
            xc_str = f" xc={xcond:.3f}" if xcond > 0 else ""
            print(f"  Ep {epoch:3d}/{EPOCHS} | NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep_metrics['recon']/n_batches:.1f} "
                  f"KL={ep_metrics['kl']/n_batches:.2f} "
                  f"τ={cfg.current_gumbel_temp:.3f} Active={n_active}{xc_str}")

    elapsed = time.time() - t_start

    # ---- 最终评估 (加载 best model) ----
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()

    final_nmi, final_acc, final_loss = evaluate_model(model, val_loader, cfg)
    final_xcond = measure_x_conditionality(model, val_loader, cfg)

    # 最终 pi
    final_pi = model.pi.detach().cpu().numpy()
    final_pi_entropy = float(-(final_pi * np.log(final_pi + 1e-9)).sum())
    final_n_active = int((final_pi > 0.02).sum())

    # 最终重构损失 (BCE, 越低越好)
    total_recon, n_eval = 0, 0
    with torch.no_grad():
        for y_img, _ in val_loader:
            y_img = y_img.to(device)
            _, info = model.forward_unlabeled(y_img, cfg)
            total_recon += info['recon']
            n_eval += 1
    final_recon = total_recon / max(n_eval, 1)

    result = {
        "config_name": cfg_dict["name"],
        "display_name": cfg_dict["display_name"],
        "seed": seed,
        "post_acc": float(final_acc),
        "nmi": float(final_nmi),
        "recon_loss": float(final_recon),
        "xcond": float(final_xcond),
        "pi_entropy": float(final_pi_entropy),
        "n_active": int(final_n_active),
        "K_model": cfg_dict["num_classes"],
        "latent_dim": cfg_dict["latent_dim"],
        "beta": cfg_dict["beta"],
        "elapsed_sec": float(elapsed),
        "history": history,
    }

    result_path = os.path.join(run_dir, "result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)

    print(f"  ✓ Done. Acc={final_acc:.4f} NMI={final_nmi:.4f} "
          f"xCond={final_xcond:.3f} Active={final_n_active} "
          f"Time={elapsed:.0f}s")
    return result


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# ==============================================================
# 批量运行
# ==============================================================
def run_all_experiments(configs_to_run=None, seeds=None,
                        save_dir="mVAE_ablation_results"):
    if configs_to_run is None:
        configs_to_run = list(ABLATION_CONFIGS.keys())
    if seeds is None:
        seeds = SEEDS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(save_dir, exist_ok=True)

    # 数据只加载一次 (用最大 batch_size 的 loader 会在每个 run 里重建)
    # 这里先用默认 cfg 拿一份 train/val split
    tmp_cfg = Config()
    tmp_cfg.batch_size = 128
    train_loader, val_loader = get_unsup_loaders(tmp_cfg)
    print(f"Data loaded: {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val")

    all_results = []
    total_runs = len(configs_to_run) * len(seeds)
    run_idx = 0

    for cfg_name in configs_to_run:
        cfg = ABLATION_CONFIGS[cfg_name]
        for seed in seeds:
            run_idx += 1
            run_name = f"{cfg_name}_seed{seed}"

            # 断点续跑
            result_path = os.path.join(save_dir, run_name, "result.json")
            if os.path.exists(result_path):
                print(f"\n[{run_idx}/{total_runs}] {run_name} — "
                      f"already done, loading.")
                with open(result_path) as f:
                    result = json.load(f)
                all_results.append(result)
                continue

            print(f"\n[{run_idx}/{total_runs}] Starting {run_name}...")

            # 为该配置重建 loader (batch_size 可能不同)
            if cfg["batch_size"] != tmp_cfg.batch_size:
                run_cfg = _make_cfg(cfg)
                run_train, run_val = get_unsup_loaders(run_cfg)
            else:
                run_train, run_val = train_loader, val_loader

            result = train_single_run(
                cfg, seed, run_train, run_val, save_dir=save_dir)
            all_results.append(result)

    # 保存汇总
    summary_path = os.path.join(save_dir, "all_results.json")
    summary = [{k: v for k, v in r.items() if k != 'history'}
               for r in all_results]
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\n✓ All results saved to {summary_path}")
    return all_results


def load_results(save_dir="mVAE_ablation_results"):
    summary_path = os.path.join(save_dir, "all_results.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    # 回退: 从各 run 目录收集
    results = []
    for d in sorted(os.listdir(save_dir)):
        rp = os.path.join(save_dir, d, "result.json")
        if os.path.exists(rp):
            with open(rp) as f:
                r = json.load(f)
                results.append({k: v for k, v in r.items() if k != 'history'})
    return results


# ==============================================================
# 结果分析与出图
# ==============================================================

# 全局样式
STYLE = {
    "fig_dpi": 200,
    "color_primary": "#2196F3",
    "color_secondary": "#4CAF50",
    "color_tertiary": "#FF9800",
    "color_accent": "#E91E63",
    "color_neutral": "#9E9E9E",
}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def generate_ablation_report(results, save_dir="mVAE_ablation_results"):
    """从结果列表生成全部对比图表和 LaTeX 表格"""
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- 聚合 ----
    grouped = defaultdict(list)
    for r in results:
        grouped[r["config_name"]].append(r)

    # 显示顺序
    ORDER = [
        "full_model",
        # 正则化消融
        "no_z_dropout", "no_balance", "no_both_reg",
        # decoder
        "decoder_concat",
        # latent_dim
        "zdim_4", "zdim_8", "zdim_16",
        # beta
        "beta_0.5", "beta_1.0", "beta_4.0",
        # K
        "K_15", "K_20",
        # tau
        "tau_aggressive", "tau_conservative",
        # 联合
        "dim8_beta1", "dim8_beta0.5", "dim16_beta0.5",
    ]
    ordered = [n for n in ORDER if n in grouped]

    # ---- 构建表格行 ----
    table_rows = []
    for name in ordered:
        runs = grouped[name]
        accs = [r["post_acc"] for r in runs]
        nmis = [r["nmi"] for r in runs]
        recons = [r["recon_loss"] for r in runs]
        xconds = [r["xcond"] for r in runs]
        actives = [r["n_active"] for r in runs]
        pi_ents = [r["pi_entropy"] for r in runs]

        table_rows.append({
            "name": name,
            "display": runs[0]["display_name"],
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "nmi_mean": np.mean(nmis), "nmi_std": np.std(nmis),
            "recon_mean": np.mean(recons), "recon_std": np.std(recons),
            "xcond_mean": np.mean(xconds), "xcond_std": np.std(xconds),
            "active_mean": np.mean(actives), "active_std": np.std(actives),
            "pi_ent_mean": np.mean(pi_ents), "pi_ent_std": np.std(pi_ents),
            "K_model": runs[0].get("K_model", 10),
            "latent_dim": runs[0].get("latent_dim", 2),
            "beta": runs[0].get("beta", 2.0),
            "n_runs": len(runs),
        })

    # ================================================================
    # 图 A1: 主指标对比 (Accuracy + Recon)
    # ================================================================
    _plot_main_comparison(table_rows, fig_dir)

    # ================================================================
    # 图 A2: x-Conditionality + Active States
    # ================================================================
    _plot_xcond_active(table_rows, fig_dir)

    # ================================================================
    # 图 A3: 正则化消融专题 (4 指标雷达/柱状)
    # ================================================================
    _plot_regularization_ablation(table_rows, grouped, fig_dir)

    # ================================================================
    # 图 A4: latent_dim × beta trade-off 热力图
    # ================================================================
    _plot_dim_beta_heatmap(table_rows, grouped, fig_dir)

    # ================================================================
    # 图 A5: latent_dim trade-off 双轴图
    # ================================================================
    _plot_dim_tradeoff(table_rows, fig_dir)

    # ================================================================
    # 图 A6: 收敛曲线对比 (从 history 加载)
    # ================================================================
    _plot_convergence_comparison(grouped, ordered, save_dir, fig_dir)

    # ================================================================
    # 图 A7: Gumbel τ 策略对比
    # ================================================================
    _plot_tau_comparison(table_rows, fig_dir)

    # ================================================================
    # LaTeX 表格 + 纯文本表格
    # ================================================================
    _generate_latex_table(table_rows, save_dir)
    _generate_text_table(table_rows, save_dir)

    return table_rows


# ----------------------------------------------------------------
# 图 A1: 主指标柱状图
# ----------------------------------------------------------------
def _plot_main_comparison(table_rows, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    labels = [r["display"] for r in table_rows]
    x = np.arange(len(labels))
    width = 0.6

    def _color(r):
        return STYLE["color_primary"] if r["name"] == "full_model" \
               else STYLE["color_neutral"]

    # (a) Posterior Accuracy
    ax = axes[0]
    vals = [r["acc_mean"] for r in table_rows]
    errs = [r["acc_std"] for r in table_rows]
    colors = [_color(r) for r in table_rows]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=3,
                  color=colors, edgecolor='white', lw=0.8)
    ax.set_ylabel("Posterior Accuracy ↑")
    ax.set_title("(a) Clustering Accuracy", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.01,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)

    # (b) Recon Loss
    ax = axes[1]
    vals = [r["recon_mean"] for r in table_rows]
    errs = [r["recon_std"] for r in table_rows]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=3,
                  color=colors, edgecolor='white', lw=0.8)
    ax.set_ylabel("Reconstruction Loss ↓")
    ax.set_title("(b) Reconstruction Quality", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.5,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    # (c) x-Conditionality
    ax = axes[2]
    vals = [r["xcond_mean"] for r in table_rows]
    errs = [r["xcond_std"] for r in table_rows]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=3,
                  color=colors, edgecolor='white', lw=0.8)
    ax.set_ylabel("x-Conditionality ↑")
    ax.set_title("(c) Decoder x-Dependence", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle("mVAE — Ablation Study: Main Metrics", fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(fig_dir, "ablation_main_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ----------------------------------------------------------------
# 图 A2: Active States + Pi Entropy
# ----------------------------------------------------------------
def _plot_xcond_active(table_rows, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r["display"] for r in table_rows]
    x = np.arange(len(labels))
    width = 0.6

    # (a) Active states
    ax = axes[0]
    vals = [r["active_mean"] for r in table_rows]
    errs_v = [r["active_std"] for r in table_rows]
    k_models = [r["K_model"] for r in table_rows]
    bars = ax.bar(x, vals, width, yerr=errs_v, capsize=3,
                  color='#4CAF50', edgecolor='white', lw=0.8)
    for i, km in enumerate(k_models):
        ax.plot([i-0.3, i+0.3], [km, km], 'r--', lw=1, alpha=0.5)
    ax.set_ylabel("# Active Classes ($\\pi_k > 0.02$)")
    ax.set_title("(a) Active Classes (red = K)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)

    # (b) Pi Entropy
    ax = axes[1]
    vals = [r["pi_ent_mean"] for r in table_rows]
    errs_v = [r["pi_ent_std"] for r in table_rows]
    k_models = [r["K_model"] for r in table_rows]
    bars = ax.bar(x, vals, width, yerr=errs_v, capsize=3,
                  color='#FF9800', edgecolor='white', lw=0.8)
    for i, km in enumerate(k_models):
        uniform_ent = np.log(km)
        ax.plot([i-0.3, i+0.3], [uniform_ent, uniform_ent],
                'g--', lw=1, alpha=0.5)
    ax.set_ylabel("$\\Pi$ Entropy")
    ax.set_title("(b) Class Prior Entropy (green = uniform)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, "ablation_active_pi.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ----------------------------------------------------------------
# 图 A3: 正则化消融专题
# ----------------------------------------------------------------
def _plot_regularization_ablation(table_rows, grouped, fig_dir):
    reg_names = ["full_model", "no_z_dropout", "no_balance", "no_both_reg"]
    reg_rows = [r for r in table_rows if r["name"] in reg_names]
    if len(reg_rows) < 2:
        return

    metrics = ["acc_mean", "xcond_mean", "recon_mean", "pi_ent_mean"]
    metric_labels = ["Post. Accuracy ↑", "x-Conditionality ↑",
                     "Recon Loss ↓", "$\\Pi$ Entropy ↑"]
    higher_better = [True, True, False, True]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    labels = [r["display"] for r in reg_rows]
    x = np.arange(len(labels))

    colors_map = {
        "full_model": "#2196F3",
        "no_z_dropout": "#FF9800",
        "no_balance": "#E91E63",
        "no_both_reg": "#9E9E9E",
    }

    for idx, (metric, label, hb) in enumerate(
            zip(metrics, metric_labels, higher_better)):
        ax = axes[idx]
        vals = [r[metric] for r in reg_rows]
        stds = [r[metric.replace("mean", "std")] for r in reg_rows]
        colors = [colors_map.get(r["name"], "#9E9E9E") for r in reg_rows]

        bars = ax.bar(x, vals, 0.6, yerr=stds, capsize=4,
                      color=colors, edgecolor='white', lw=0.8)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)

        # 标注 full model 的值
        for bar, v, s in zip(bars, vals, stds):
            fmt = f'{v:.1%}' if 'acc' in metric or 'xcond' in metric else f'{v:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
                    fmt, ha='center', va='bottom', fontsize=8)

    fig.suptitle("mVAE — Regularization Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, "ablation_regularization.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ----------------------------------------------------------------
# 图 A4: latent_dim × beta 热力图
# ----------------------------------------------------------------
def _plot_dim_beta_heatmap(table_rows, grouped, fig_dir):
    # 收集所有 (dim, beta) → acc 数据
    dim_beta_data = {}
    for r in table_rows:
        key = (r["latent_dim"], r["beta"])
        dim_beta_data[key] = r

    dims = sorted(set(k[0] for k in dim_beta_data.keys()))
    betas = sorted(set(k[1] for k in dim_beta_data.keys()))

    if len(dims) < 2 or len(betas) < 2:
        return

    # 构建热力矩阵
    for metric, label, cmap in [
        ("acc_mean", "Posterior Accuracy", "YlOrRd"),
        ("xcond_mean", "x-Conditionality", "YlGnBu"),
    ]:
        mat = np.full((len(betas), len(dims)), np.nan)
        for bi, b in enumerate(betas):
            for di, d in enumerate(dims):
                if (d, b) in dim_beta_data:
                    mat[bi, di] = dim_beta_data[(d, b)][metric]

        if np.isnan(mat).all():
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, cmap=cmap, aspect='auto',
                        vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([str(b) for b in betas])
        ax.set_xlabel("$d_z$ (latent dim)")
        ax.set_ylabel("$\\beta$ (KL weight)")
        ax.set_title(f"mVAE — {label} ($d_z \\times \\beta$)",
                     fontweight='bold')

        for bi in range(len(betas)):
            for di in range(len(dims)):
                if not np.isnan(mat[bi, di]):
                    fmt = f'{mat[bi,di]:.1%}' if 'acc' in metric else f'{mat[bi,di]:.3f}'
                    ax.text(di, bi, fmt, ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if mat[bi,di] > np.nanmedian(mat) else 'black')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        safe_metric = metric.replace("_mean", "")
        path = os.path.join(fig_dir, f"ablation_heatmap_{safe_metric}.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {path}")


# ----------------------------------------------------------------
# 图 A5: latent_dim trade-off 双轴图
# ----------------------------------------------------------------
def _plot_dim_tradeoff(table_rows, fig_dir):
    dim_names = ["full_model", "zdim_4", "zdim_8", "zdim_16"]
    dim_rows = [r for r in table_rows if r["name"] in dim_names]
    if len(dim_rows) < 2:
        return

    dim_rows_sorted = sorted(dim_rows, key=lambda r: r["latent_dim"])
    dims = [r["latent_dim"] for r in dim_rows_sorted]
    accs = [r["acc_mean"] for r in dim_rows_sorted]
    acc_stds = [r["acc_std"] for r in dim_rows_sorted]
    xconds = [r["xcond_mean"] for r in dim_rows_sorted]
    xc_stds = [r["xcond_std"] for r in dim_rows_sorted]
    recons = [r["recon_mean"] for r in dim_rows_sorted]
    rec_stds = [r["recon_std"] for r in dim_rows_sorted]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    c1, c2, c3 = '#2196F3', '#E91E63', '#4CAF50'

    ax1.errorbar(dims, accs, yerr=acc_stds, marker='o', color=c1,
                 lw=2, capsize=5, label='Post. Accuracy ↑')
    ax1.errorbar(dims, xconds, yerr=xc_stds, marker='D', color=c3,
                 lw=2, capsize=5, label='x-Conditionality ↑')
    ax1.set_xlabel("$d_z$ (latent dim)", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12, color='black')
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(dims)
    ax1.set_xticklabels([str(d) for d in dims])

    ax2 = ax1.twinx()
    ax2.errorbar(dims, recons, yerr=rec_stds, marker='s', color=c2,
                 lw=2, capsize=5, ls='--', label='Recon Loss ↓')
    ax2.set_ylabel("Recon Loss", fontsize=12, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.set_title("$d_z$ Trade-off: Accuracy / x-Cond / Recon",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, "ablation_dim_tradeoff.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ----------------------------------------------------------------
# 图 A6: 收敛曲线对比
# ----------------------------------------------------------------
def _plot_convergence_comparison(grouped, ordered, save_dir, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.tab20

    for idx, name in enumerate(ordered):
        # 加载第一个 seed 的 history
        run_dir = os.path.join(save_dir, f"{name}_seed{SEEDS[0]}")
        hist_path = os.path.join(run_dir, "result.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            full_r = json.load(f)
        hist = full_r.get("history", [])
        if not hist:
            continue

        epochs = [h["epoch"] for h in hist]
        color = cmap(idx / max(len(ordered), 1))
        display = grouped[name][0]["display_name"]

        # (a) Posterior Accuracy
        acc_curve = [h.get("post_acc", 0) for h in hist]
        axes[0].plot(epochs, acc_curve, lw=1.2, alpha=0.8,
                     color=color, label=display)

        # (b) Recon Loss
        recon_curve = [h.get("recon", 0) for h in hist]
        axes[1].plot(epochs, recon_curve, lw=1.2, alpha=0.8,
                     color=color, label=display)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Posterior Accuracy")
    axes[0].set_title("(a) Accuracy Convergence", fontweight='bold')
    axes[0].legend(fontsize=7, loc='lower right', ncol=2)
    axes[0].set_ylim(0, 1)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recon Loss")
    axes[1].set_title("(b) Reconstruction Convergence", fontweight='bold')
    axes[1].legend(fontsize=7, loc='upper right', ncol=2)

    fig.suptitle("mVAE — Convergence Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, "ablation_convergence.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ----------------------------------------------------------------
# 图 A7: Gumbel τ 策略对比
# ----------------------------------------------------------------
def _plot_tau_comparison(table_rows, fig_dir):
    tau_names = ["full_model", "tau_aggressive", "tau_conservative"]
    tau_rows = [r for r in table_rows if r["name"] in tau_names]
    if len(tau_rows) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [r["display"] for r in tau_rows]
    x = np.arange(len(labels))
    colors = ['#2196F3', '#E91E63', '#FF9800']

    for idx, (metric, ylabel, title) in enumerate([
        ("acc_mean", "Post. Accuracy", "(a) Accuracy"),
        ("xcond_mean", "x-Conditionality", "(b) x-Cond"),
        ("pi_ent_mean", "$\\Pi$ Entropy", "(c) Pi Entropy"),
    ]):
        ax = axes[idx]
        vals = [r[metric] for r in tau_rows]
        stds = [r[metric.replace("mean", "std")] for r in tau_rows]
        bars = ax.bar(x, vals, 0.5, yerr=stds, capsize=4,
                      color=colors[:len(tau_rows)], edgecolor='white')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

    fig.suptitle("mVAE — Gumbel $\\tau$ Strategy Comparison",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, "ablation_tau_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


# ----------------------------------------------------------------
# LaTeX 表格
# ----------------------------------------------------------------
def _generate_latex_table(table_rows, save_dir):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study on mVAE (unsupervised MNIST, 3 seeds).}",
        r"\label{tab:mvae_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuration & Accuracy $\uparrow$ & NMI $\uparrow$ "
        r"& Recon $\downarrow$ & x-Cond $\uparrow$ & Active \\",
        r"\midrule",
    ]
    for r in table_rows:
        name = r["display"].replace("_", r"\_")
        if r["name"] == "full_model":
            name = r"\textbf{" + name + "}"

        acc = f"${r['acc_mean']:.1%} \\pm {r['acc_std']:.1%}$".replace("%", r"\%")
        nmi = f"${r['nmi_mean']:.4f} \\pm {r['nmi_std']:.4f}$"
        rec = f"${r['recon_mean']:.1f} \\pm {r['recon_std']:.1f}$"
        xc = f"${r['xcond_mean']:.3f} \\pm {r['xcond_std']:.3f}$"
        act = f"${r['active_mean']:.1f}/{r['K_model']}$"
        lines.append(f"  {name} & {acc} & {nmi} & {rec} & {xc} & {act} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]

    path = os.path.join(save_dir, "ablation_table.tex")
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table → {path}")


def _generate_text_table(table_rows, save_dir):
    path = os.path.join(save_dir, "ablation_table.txt")
    with open(path, 'w') as f:
        header = (f"{'Configuration':<28s} {'Accuracy':>16s} {'NMI':>14s} "
                  f"{'Recon':>14s} {'x-Cond':>14s} {'Active':>8s}")
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")
        for r in table_rows:
            f.write(
                f"{r['display']:<28s} "
                f"{r['acc_mean']:.1%} ± {r['acc_std']:.1%}   "
                f"{r['nmi_mean']:.4f} ± {r['nmi_std']:.4f}   "
                f"{r['recon_mean']:.1f} ± {r['recon_std']:.1f}   "
                f"{r['xcond_mean']:.3f} ± {r['xcond_std']:.3f}   "
                f"{r['active_mean']:.1f}/{r['K_model']}\n"
            )
    print(f"  Text table → {path}")

    # 打印到控制台
    print(f"\n{'='*80}")
    print("mVAE ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    with open(path) as f:
        print(f.read())


# ==============================================================
# CLI
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mVAE Ablation Study")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="mVAE_ablation_results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 30 epochs for fast iteration")
    args = parser.parse_args()

    if args.quick:
        for cfg in ABLATION_CONFIGS.values():
            cfg["epochs"] = 30

    if args.plot_only:
        results = load_results(args.save_dir)
        if not results:
            print("No results found!")
        else:
            generate_ablation_report(results, args.save_dir)
    else:
        configs = [args.config] if args.config else None
        seeds = [args.seed] if args.seed else None
        results = run_all_experiments(
            configs_to_run=configs, seeds=seeds, save_dir=args.save_dir)
        generate_ablation_report(results, args.save_dir)
