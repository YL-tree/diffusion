"""
mVAE_ablation.py — mVAE 消融实验自动化框架 (unsup + semisup 双模式)
====================================================================

两阶段方法论:
  Phase 0: Optuna 搜索 → 找到最优 baseline 超参 (只跑一次)
  Phase 1: 消融实验   → 固定 baseline, 控制变量 (×3 seeds)

功能:
  1. [Phase 0] Optuna 搜索 baseline (可选, --optuna)
  2. [Phase 1] 定义 unsup (18) + semisup (6) 共 24 个消融配置
  3. 自动运行全部实验 (每个配置 x 3 seeds, 断点续跑)
  4. 生成对比图表和 LaTeX 表格

使用:
  # Phase 0: Optuna 找 baseline (建议先跑这步)
  python mVAE_ablation.py --optuna --mode unsup            # 100 trials
  python mVAE_ablation.py --optuna --mode semisup --n-trials 50

  # Phase 1: 消融实验 (自动加载 Optuna 找到的 baseline)
  python mVAE_ablation.py                                  # unsup 全部
  python mVAE_ablation.py --mode semisup                   # semisup 全部
  python mVAE_ablation.py --mode both                      # 两者都跑
  python mVAE_ablation.py --plot-only                      # 仅出图
  python mVAE_ablation.py --config full_model --seed 42    # 单个配置
  python mVAE_ablation.py --quick                          # 30 epochs

依赖: mVAE_aligned.py, mVAE_common.py, optuna (Phase 0 only)
"""

import os, sys, json, time, argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mVAE_common import Config, compute_NMI, compute_posterior_accuracy
from mVAE_aligned import (
    mVAE, get_unsup_loaders, get_semisup_loaders,
    evaluate_model, measure_x_conditionality,
)


# ==============================================================
# Phase 0: Optuna Baseline Search
# ==============================================================

OPTUNA_SEARCH_SPACE = {
    # 参数名:  (type, choices_or_range)
    "latent_dim":     ("categorical", [2, 4, 8, 16]),
    "beta":           ("categorical", [0.5, 1.0, 2.0, 4.0]),
    "z_dropout_rate":  ("categorical", [0.0, 0.3, 0.5, 0.7]),
    "balance_weight":  ("categorical", [0.0, 1.0, 5.0, 10.0, 20.0]),
    "tau_min":         ("categorical", [0.05, 0.1, 0.2, 0.3]),
    "tau_rate":        ("categorical", [0.95, 0.97, 0.98, 0.99]),
    "lr":              ("loguniform",  [5e-4, 3e-3]),
}

# 不搜索的参数 (固定)
OPTUNA_FIXED = dict(
    num_classes=10, decoder_type="film", batch_size=128,
    tau_start=1.0, epochs=15,  # 短 epochs 做快速筛选
    alpha_unlabeled=0.5, labeled_per_class=100,
)


def _optuna_objective(trial, mode, loaders):
    """Optuna 目标函数: 复合得分 = 0.6*acc + 0.2*xcond + 0.2*recon_score"""
    import warnings
    # ★★ 抑制 KMeans 对坍缩 z 空间的 ConvergenceWarning
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    params = {}
    for name, (ptype, spec) in OPTUNA_SEARCH_SPACE.items():
        if ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec)
        elif ptype == "loguniform":
            params[name] = trial.suggest_float(name, spec[0], spec[1], log=True)

    cfg_dict = {**OPTUNA_FIXED, **params, "name": "optuna_trial",
                "display_name": "optuna_trial"}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = _make_cfg(cfg_dict, device)

    torch.manual_seed(42)
    np.random.seed(42)
    model = mVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    EPOCHS = cfg_dict["epochs"]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        cfg.current_gumbel_temp = max(
            cfg.min_gumbel_temp,
            cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        if mode == "unsup":
            _train_epoch_unsup(model, optimizer, loaders["train"], cfg)
        else:
            _train_epoch_semisup(model, optimizer,
                                 loaders["labeled"], loaders["unlabeled"], cfg)

        # ★★ 退化检测 + Pruning: 每 5 epoch 检查
        if epoch % 5 == 0:
            _, mid_acc, _ = evaluate_model(model, loaders["val"], cfg)

            # 退化: epoch 5 后 acc 还 < 15% (随机水平 = 10%) → 直接剪
            if epoch >= 5 and mid_acc < 0.15:
                import optuna
                raise optuna.TrialPruned(f"Degenerate: acc={mid_acc:.3f} at ep {epoch}")

            # 坍缩检测: pi 最大分量 > 0.8 说明只用了 1-2 个类
            pi_np = model.pi.detach().cpu().numpy()
            if pi_np.max() > 0.8:
                import optuna
                raise optuna.TrialPruned(f"Collapsed: pi_max={pi_np.max():.3f}")

            trial.report(mid_acc, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    # 最终评估
    _, post_acc, _ = evaluate_model(model, loaders["val"], cfg)
    xcond = measure_x_conditionality(model, loaders["val"], cfg)

    # 重构损失 (归一化到 [0,1])
    total_recon, n = 0, 0
    with torch.no_grad():
        for y, _ in loaders["val"]:
            y = y.to(device)
            _, info = model.forward_unlabeled(y, cfg)
            total_recon += info['recon']; n += 1
    recon = total_recon / max(n, 1)
    # 用 baseline_recon=200 做归一化 (MNIST 随机初始化约 200-250)
    recon_score = max(0, 1 - recon / 200.0)

    composite = 0.6 * post_acc + 0.2 * xcond + 0.2 * recon_score
    return composite


def run_optuna_search(mode="unsup", n_trials=100, save_dir=None):
    """
    Phase 0: 用 Optuna 搜索最优 baseline 超参.
    结果保存为 JSON, 后续消融实验自动加载.
    """
    import optuna
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    sd = save_dir or f"mVAE_ablation_{mode}"
    os.makedirs(sd, exist_ok=True)
    result_path = os.path.join(sd, "optuna_best.json")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmp_cfg = _make_cfg({**OPTUNA_FIXED, **{k: v[1][0] if v[0] == "categorical"
                                             else v[1][0]
                                             for k, v in OPTUNA_SEARCH_SPACE.items()},
                         "name": "tmp", "display_name": "tmp"}, device)
    loaders = _load_data(mode, tmp_cfg)

    print(f"\n{'#'*60}")
    print(f"# Phase 0: Optuna Search ({mode})")
    print(f"# {n_trials} trials, {OPTUNA_FIXED['epochs']} epochs each")
    print(f"# Search space: {list(OPTUNA_SEARCH_SPACE.keys())}")
    print(f"{'#'*60}\n")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"mVAE_{mode}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: _optuna_objective(trial, mode, loaders),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Optuna best trial #{best.number}")
    print(f"  Score:  {best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=2)}")
    print(f"{'='*60}")

    # 保存
    result = {
        "mode": mode,
        "best_score": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
        "fixed_params": OPTUNA_FIXED,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved → {result_path}")
    print(f"\n下一步: 运行 python mVAE_ablation.py --mode {mode}")
    print(f"消融实验会自动加载这些最优参数作为 baseline.")

    return best.params


def _load_optuna_baseline(save_dir, mode):
    """尝试加载 Phase 0 的 Optuna 结果, 返回 dict 或 None."""
    path = os.path.join(save_dir, "optuna_best.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data.get("mode") == mode:
            print(f"  ★ Loaded Optuna baseline from {path}")
            print(f"    Score: {data['best_score']:.4f}")
            print(f"    Params: {data['best_params']}")
            return data["best_params"]
    return None


# ==============================================================
# 配置定义
# ==============================================================

# 默认 _BASE (可被 Optuna 结果覆盖)
_BASE = dict(
    latent_dim=2, num_classes=10, beta=2.0,
    z_dropout_rate=0.5, balance_weight=10.0,
    decoder_type="film", lr=1e-3, batch_size=128,
    epochs=100,
    tau_start=1.0, tau_min=0.1, tau_rate=0.98,
    # 半监督专用 (unsup 时忽略)
    alpha_unlabeled=0.5, labeled_per_class=100,
)

def _c(name, display, **kw):
    d = {**_BASE, "name": name, "display_name": display}
    d.update(kw)
    return d

# ---- 无监督消融 (18) ----
UNSUP_CONFIGS = {
    "full_model":       _c("full_model",       "Full model"),

    "zdim_4":           _c("zdim_4",           "$d_z=4$",              latent_dim=4),
    "zdim_8":           _c("zdim_8",           "$d_z=8$",              latent_dim=8),
    "zdim_16":          _c("zdim_16",          "$d_z=16$",             latent_dim=16),

    "beta_0.5":         _c("beta_0.5",         "$\\beta=0.5$",         beta=0.5),
    "beta_1.0":         _c("beta_1.0",         "$\\beta=1.0$",         beta=1.0),
    "beta_4.0":         _c("beta_4.0",         "$\\beta=4.0$",         beta=4.0),

    "no_z_dropout":     _c("no_z_dropout",     "w/o z-dropout",        z_dropout_rate=0.0),
    "no_balance":       _c("no_balance",       "w/o balance",          balance_weight=0.0),
    "no_both_reg":      _c("no_both_reg",      "w/o both reg.",        z_dropout_rate=0.0,
                                                                        balance_weight=0.0),

    "decoder_concat":   _c("decoder_concat",   "Concat decoder",       decoder_type="concat"),

    "K_15":             _c("K_15",             "$K=15$",               num_classes=15),
    "K_20":             _c("K_20",             "$K=20$",               num_classes=20),

    "tau_aggressive":   _c("tau_aggressive",   "$\\tau$: aggressive",  tau_min=0.05, tau_rate=0.95),
    "tau_conservative": _c("tau_conservative", "$\\tau$: conservative", tau_min=0.3, tau_rate=0.99),

    "dim8_beta1":       _c("dim8_beta1",       "$d_z=8,\\beta=1$",     latent_dim=8, beta=1.0),
    "dim8_beta0.5":     _c("dim8_beta0.5",     "$d_z=8,\\beta=0.5$",   latent_dim=8, beta=0.5),
    "dim16_beta0.5":    _c("dim16_beta0.5",    "$d_z=16,\\beta=0.5$",  latent_dim=16, beta=0.5),
}

# ---- 半监督消融 (6) ----
SEMISUP_CONFIGS = {
    "ss_full":          _c("ss_full",          "SemiSup full"),

    "ss_alpha_0.1":     _c("ss_alpha_0.1",     "$\\alpha_{un}=0.1$",   alpha_unlabeled=0.1),
    "ss_alpha_1.0":     _c("ss_alpha_1.0",     "$\\alpha_{un}=1.0$",   alpha_unlabeled=1.0),

    "ss_label_10":      _c("ss_label_10",      "10 labels/class",      labeled_per_class=10),
    "ss_label_50":      _c("ss_label_50",      "50 labels/class",      labeled_per_class=50),
    "ss_label_500":     _c("ss_label_500",     "500 labels/class",     labeled_per_class=500),
}

SEEDS = [42, 123, 2024]


def _rebuild_configs_from_base(base):
    """
    用给定的 base dict 重新生成所有消融配置.
    当 Optuna 找到更优的 baseline 时, 调用此函数更新.
    """
    def _c(name, display, **kw):
        d = {**base, "name": name, "display_name": display}
        d.update(kw)
        return d

    unsup = {
        "full_model":       _c("full_model",       "Full model"),
        "zdim_4":           _c("zdim_4",           "$d_z=4$",              latent_dim=4),
        "zdim_8":           _c("zdim_8",           "$d_z=8$",              latent_dim=8),
        "zdim_16":          _c("zdim_16",          "$d_z=16$",             latent_dim=16),
        "beta_0.5":         _c("beta_0.5",         "$\\beta=0.5$",         beta=0.5),
        "beta_1.0":         _c("beta_1.0",         "$\\beta=1.0$",         beta=1.0),
        "beta_4.0":         _c("beta_4.0",         "$\\beta=4.0$",         beta=4.0),
        "no_z_dropout":     _c("no_z_dropout",     "w/o z-dropout",        z_dropout_rate=0.0),
        "no_balance":       _c("no_balance",       "w/o balance",          balance_weight=0.0),
        "no_both_reg":      _c("no_both_reg",      "w/o both reg.",        z_dropout_rate=0.0,
                                                                            balance_weight=0.0),
        "decoder_concat":   _c("decoder_concat",   "Concat decoder",       decoder_type="concat"),
        "K_15":             _c("K_15",             "$K=15$",               num_classes=15),
        "K_20":             _c("K_20",             "$K=20$",               num_classes=20),
        "tau_aggressive":   _c("tau_aggressive",   "$\\tau$: aggressive",  tau_min=0.05, tau_rate=0.95),
        "tau_conservative": _c("tau_conservative", "$\\tau$: conservative", tau_min=0.3, tau_rate=0.99),
        "dim8_beta1":       _c("dim8_beta1",       "$d_z=8,\\beta=1$",     latent_dim=8, beta=1.0),
        "dim8_beta0.5":     _c("dim8_beta0.5",     "$d_z=8,\\beta=0.5$",   latent_dim=8, beta=0.5),
        "dim16_beta0.5":    _c("dim16_beta0.5",    "$d_z=16,\\beta=0.5$",  latent_dim=16, beta=0.5),
    }

    semisup = {
        "ss_full":          _c("ss_full",          "SemiSup full"),
        "ss_alpha_0.1":     _c("ss_alpha_0.1",     "$\\alpha_{un}=0.1$",   alpha_unlabeled=0.1),
        "ss_alpha_1.0":     _c("ss_alpha_1.0",     "$\\alpha_{un}=1.0$",   alpha_unlabeled=1.0),
        "ss_label_10":      _c("ss_label_10",      "10 labels/class",      labeled_per_class=10),
        "ss_label_50":      _c("ss_label_50",      "50 labels/class",      labeled_per_class=50),
        "ss_label_500":     _c("ss_label_500",     "500 labels/class",     labeled_per_class=500),
    }
    return unsup, semisup


# ==============================================================
# config dict → Config 对象
# ==============================================================
def _make_cfg(d, device='cpu'):
    cfg = Config()
    cfg.latent_dim         = d["latent_dim"]
    cfg.num_classes        = d["num_classes"]
    cfg.beta               = d["beta"]
    cfg.z_dropout_rate     = d["z_dropout_rate"]
    cfg.balance_weight     = d["balance_weight"]
    cfg.decoder_type       = d["decoder_type"]
    cfg.lr                 = d["lr"]
    cfg.batch_size         = d["batch_size"]
    cfg.final_epochs       = d["epochs"]
    cfg.init_gumbel_temp   = d["tau_start"]
    cfg.min_gumbel_temp    = d["tau_min"]
    cfg.gumbel_anneal_rate = d["tau_rate"]
    cfg.current_gumbel_temp= d["tau_start"]
    cfg.alpha_unlabeled    = d.get("alpha_unlabeled", 0.5)
    cfg.labeled_per_class  = d.get("labeled_per_class", 100)
    cfg.device             = device
    cfg.output_dir         = "./tmp_ablation"
    return cfg


# ==============================================================
# 单 epoch 训练: unsup / semisup
# ==============================================================
def _train_epoch_unsup(model, optimizer, train_loader, cfg):
    """无监督: 遍历 train_loader, 每 batch 调 forward_unlabeled."""
    model.train()
    ep = defaultdict(float)
    n = 0
    for y_img, _ in train_loader:
        y_img = y_img.to(cfg.device)
        loss, info = model.forward_unlabeled(y_img, cfg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        for k in ['recon', 'kl', 'prior', 'post_corr', 'resp_ent', 'balance']:
            ep[k] += info.get(k, 0)
        n += 1
    return {k: v / max(n, 1) for k, v in ep.items()}, n


def _train_epoch_semisup(model, optimizer, labeled_loader, unlabeled_loader, cfg):
    """
    半监督: 遍历全部 unlabeled, labeled 循环复用.
    与 mVAE_aligned.train_semisupervised 内循环一致:
      loss = loss_lab + α_unlabeled * loss_un
    """
    model.train()
    ep = defaultdict(float)
    n = 0
    labeled_iter = iter(labeled_loader)

    for x_un, _ in unlabeled_loader:
        # labeled 用完则重建 iter
        try:
            x_lab, y_lab = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_lab, y_lab = next(labeled_iter)

        x_lab = x_lab.to(cfg.device)
        y_lab = y_lab.to(cfg.device)
        x_un  = x_un.to(cfg.device)

        loss_lab, info_lab = model.forward_labeled(x_lab, y_lab, cfg)
        loss_un,  info_un  = model.forward_unlabeled(x_un, cfg)
        loss = loss_lab + cfg.alpha_unlabeled * loss_un

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # 与 train_semisupervised 相同的聚合方式
        ep['recon']    += (info_lab['recon'] + info_un['recon']) / 2
        ep['kl']       += (info_lab['kl']    + info_un['kl'])    / 2
        ep['prior']    += info_lab['prior']
        ep['post_corr']+= info_un['post_corr']
        ep['resp_ent'] += info_un['resp_ent']
        ep['balance']  += info_un['balance']
        n += 1

    return {k: v / max(n, 1) for k, v in ep.items()}, n


# ==============================================================
# 训练入口
# ==============================================================
def train_single_run(config, seed, mode, loaders, save_dir):
    """
    训练一次 mVAE, 返回评估指标 dict.

    参数:
      config:  配置 dict (来自 UNSUP_CONFIGS 或 SEMISUP_CONFIGS)
      seed:    随机种子
      mode:    "unsup" | "semisup"
      loaders: dict
               unsup   → {"train": DataLoader, "val": DataLoader}
               semisup → {"labeled": DL, "unlabeled": DL, "val": DL}
      save_dir: 结果保存目录
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg_dict = config
    run_name = f"{cfg_dict['name']}_seed{seed}"
    run_dir  = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = _make_cfg(cfg_dict, device)

    print(f"\n{'='*60}")
    print(f"  [{mode.upper()}] {cfg_dict['display_name']}  |  seed={seed}")
    print(f"  d_z={cfg.latent_dim}, K={cfg.num_classes}, β={cfg.beta}, "
          f"z_drop={cfg.z_dropout_rate}, bal={cfg.balance_weight}")
    if mode == "semisup":
        print(f"  α_un={cfg.alpha_unlabeled}, labels/class={cfg.labeled_per_class}")
    print(f"{'='*60}")

    model     = mVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    val_loader = loaders["val"]

    EPOCHS = cfg_dict["epochs"]
    best_acc = 0.0
    best_model_path = os.path.join(run_dir, "best_model.pt")
    history = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        cfg.current_gumbel_temp = max(
            cfg.min_gumbel_temp,
            cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        # ======= 按 mode 分发训练 =======
        if mode == "unsup":
            ep_avg, _ = _train_epoch_unsup(
                model, optimizer, loaders["train"], cfg)
        else:
            ep_avg, _ = _train_epoch_semisup(
                model, optimizer,
                loaders["labeled"], loaders["unlabeled"], cfg)

        # 评估
        nmi, post_acc, val_loss = evaluate_model(model, val_loader, cfg)
        if post_acc > best_acc:
            best_acc = post_acc
            torch.save(model.state_dict(), best_model_path)

        xcond = 0.0
        if epoch % 20 == 0 or epoch == EPOCHS:
            xcond = measure_x_conditionality(model, val_loader, cfg)

        pi_np       = model.pi.detach().cpu().numpy()
        pi_entropy  = float(-(pi_np * np.log(pi_np + 1e-9)).sum())
        n_active    = int((pi_np > 0.02).sum())

        history.append(dict(
            epoch=epoch, loss=val_loss,
            recon=ep_avg.get('recon', 0), kl=ep_avg.get('kl', 0),
            prior=ep_avg.get('prior', 0), post_corr=ep_avg.get('post_corr', 0),
            resp_ent=ep_avg.get('resp_ent', 0), balance=ep_avg.get('balance', 0),
            nmi=nmi, post_acc=post_acc,
            xcond=xcond if xcond > 0 else None,
            pi_entropy=pi_entropy, n_active=n_active,
            tau=cfg.current_gumbel_temp, pi_values=pi_np.tolist(),
        ))

        if epoch % 25 == 0 or epoch == EPOCHS:
            xc = f" xc={xcond:.3f}" if xcond > 0 else ""
            print(f"  Ep {epoch:3d}/{EPOCHS} | NMI={nmi:.4f} Acc={post_acc:.4f} "
                  f"| R={ep_avg.get('recon',0):.1f} KL={ep_avg.get('kl',0):.2f} "
                  f"τ={cfg.current_gumbel_temp:.3f} Act={n_active}{xc}")

    elapsed = time.time() - t0

    # ---- 最终评估 (best model) ----
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()
    final_nmi, final_acc, _ = evaluate_model(model, val_loader, cfg)
    final_xcond = measure_x_conditionality(model, val_loader, cfg)

    final_pi = model.pi.detach().cpu().numpy()

    total_recon, n_eval = 0, 0
    with torch.no_grad():
        for y_img, _ in val_loader:
            y_img = y_img.to(device)
            _, info = model.forward_unlabeled(y_img, cfg)
            total_recon += info['recon']
            n_eval += 1

    result = dict(
        config_name=cfg_dict["name"],
        display_name=cfg_dict["display_name"],
        mode=mode, seed=seed,
        post_acc=float(final_acc), nmi=float(final_nmi),
        recon_loss=float(total_recon / max(n_eval, 1)),
        xcond=float(final_xcond),
        pi_entropy=float(-(final_pi * np.log(final_pi + 1e-9)).sum()),
        n_active=int((final_pi > 0.02).sum()),
        K_model=cfg_dict["num_classes"],
        latent_dim=cfg_dict["latent_dim"],
        beta=cfg_dict["beta"],
        alpha_unlabeled=cfg_dict.get("alpha_unlabeled", 0.5),
        labeled_per_class=cfg_dict.get("labeled_per_class", 100),
        elapsed_sec=float(elapsed),
        history=history,
    )

    with open(os.path.join(run_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2, default=_jdefault)

    print(f"  ✓ Acc={final_acc:.4f} NMI={final_nmi:.4f} "
          f"xCond={final_xcond:.3f} Act={result['n_active']} "
          f"Time={elapsed:.0f}s")
    return result


def _jdefault(o):
    if isinstance(o, np.ndarray):                return o.tolist()
    if isinstance(o, (np.float32, np.float64)):  return float(o)
    if isinstance(o, (np.int32,   np.int64)):    return int(o)
    return o


# ==============================================================
# 数据加载
# ==============================================================
def _load_data(mode, cfg):
    """返回 loaders dict, 根据 mode 决定调 get_unsup/get_semisup."""
    if mode == "unsup":
        tr, val = get_unsup_loaders(cfg)
        return {"train": tr, "val": val}
    else:
        lab, unlab, val = get_semisup_loaders(cfg)
        return {"labeled": lab, "unlabeled": unlab, "val": val}


# ==============================================================
# 批量运行
# ==============================================================
def run_all_experiments(mode="unsup", configs_to_run=None, seeds=None,
                        save_dir=None, quick_epochs=None):
    """
    Phase 1: 运行消融实验.
    自动检查是否有 Phase 0 的 Optuna 结果, 有则更新 baseline.

    mode: "unsup" | "semisup" | "both"
    quick_epochs: 若非 None, 强制所有配置用此 epoch 数
    """
    global UNSUP_CONFIGS, SEMISUP_CONFIGS

    if seeds is None:
        seeds = SEEDS
    modes = ["unsup", "semisup"] if mode == "both" else [mode]
    all_results = []

    for m in modes:
        sd = save_dir or f"mVAE_ablation_{m}"
        os.makedirs(sd, exist_ok=True)

        # ★★ 尝试加载 Optuna baseline → 重建所有配置
        optuna_params = _load_optuna_baseline(sd, m)
        if optuna_params:
            updated_base = {**_BASE}
            for k, v in optuna_params.items():
                if k in updated_base:
                    updated_base[k] = v
            new_unsup, new_semisup = _rebuild_configs_from_base(updated_base)
            UNSUP_CONFIGS = new_unsup
            SEMISUP_CONFIGS = new_semisup
            print(f"  ★ Configs rebuilt with Optuna baseline:")
            print(f"    d_z={updated_base['latent_dim']}, β={updated_base['beta']}, "
                  f"z_drop={updated_base['z_dropout_rate']}, "
                  f"bal={updated_base['balance_weight']}, lr={updated_base['lr']:.4f}")
        else:
            print(f"  (No Optuna baseline found in {sd}, using defaults)")

        pool = UNSUP_CONFIGS if m == "unsup" else SEMISUP_CONFIGS

        # quick 模式: 在 Optuna rebuild 之后强制覆盖 epochs
        if quick_epochs is not None:
            for c in pool.values():
                c["epochs"] = quick_epochs

        names = configs_to_run or list(pool.keys())
        names = [n for n in names if n in pool]
        if not names:
            print(f"[{m}] No matching configs — skip"); continue

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"\n{'#'*60}")
        print(f"# {m.upper()} | {len(names)} configs × {len(seeds)} seeds | {device}")
        print(f"# → {sd}")
        print(f"{'#'*60}")

        # 默认 loader (大部分配置共享)
        default_cfg = _make_cfg(list(pool.values())[0], device)
        default_loaders = _load_data(m, default_cfg)

        total = len(names) * len(seeds)
        idx = 0
        for cname in names:
            cd = pool[cname]
            for seed in seeds:
                idx += 1
                rn = f"{cname}_seed{seed}"
                rp = os.path.join(sd, rn, "result.json")

                if os.path.exists(rp):
                    print(f"\n[{idx}/{total}] {rn} — cached")
                    with open(rp) as f:
                        all_results.append(json.load(f))
                    continue

                # 需要重建 loader 的条件
                need_reload = (
                    cd["batch_size"]   != default_cfg.batch_size or
                    cd["num_classes"]  != default_cfg.num_classes or
                    (m == "semisup" and
                     cd.get("labeled_per_class") != default_cfg.labeled_per_class)
                )
                loaders = (_load_data(m, _make_cfg(cd, device))
                           if need_reload else default_loaders)

                print(f"\n[{idx}/{total}] {rn}")
                r = train_single_run(cd, seed, m, loaders, save_dir=sd)
                all_results.append(r)

        # 汇总
        sp = os.path.join(sd, "all_results.json")
        mode_r = [r for r in all_results if r.get("mode") == m]
        with open(sp, 'w') as f:
            json.dump([{k: v for k, v in r.items() if k != 'history'}
                       for r in mode_r], f, indent=2, default=_jdefault)
        print(f"\n✓ [{m}] saved → {sp}")

    return all_results


def load_results(sd):
    sp = os.path.join(sd, "all_results.json")
    if os.path.exists(sp):
        with open(sp) as f: return json.load(f)
    out = []
    for d in sorted(os.listdir(sd)):
        rp = os.path.join(sd, d, "result.json")
        if os.path.exists(rp):
            with open(rp) as f:
                r = json.load(f)
                out.append({k: v for k, v in r.items() if k != 'history'})
    return out


# ==============================================================
# 出图 & 表格
# ==============================================================

STYLE = dict(
    primary="#2196F3", secondary="#4CAF50",
    tertiary="#FF9800", accent="#E91E63", neutral="#9E9E9E",
)
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def generate_ablation_report(results, save_dir, mode="unsup"):
    fig_dir = os.path.join(save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    grouped = defaultdict(list)
    for r in results:
        grouped[r["config_name"]].append(r)

    if mode == "unsup":
        order = [
            "full_model",
            "no_z_dropout", "no_balance", "no_both_reg", "decoder_concat",
            "zdim_4", "zdim_8", "zdim_16",
            "beta_0.5", "beta_1.0", "beta_4.0",
            "K_15", "K_20",
            "tau_aggressive", "tau_conservative",
            "dim8_beta1", "dim8_beta0.5", "dim16_beta0.5",
        ]
    else:
        order = [
            "ss_full",
            "ss_alpha_0.1", "ss_alpha_1.0",
            "ss_label_10", "ss_label_50", "ss_label_500",
        ]
    ordered = [n for n in order if n in grouped]
    rows = _build_rows(grouped, ordered)

    # 通用图
    _fig_main(rows, fig_dir, mode)
    _fig_active_pi(rows, fig_dir, mode)
    _fig_convergence(grouped, ordered, save_dir, fig_dir, mode)

    # unsup 专属
    if mode == "unsup":
        _fig_reg_ablation(rows, fig_dir)
        _fig_dim_beta_heatmap(rows, fig_dir)
        _fig_dim_tradeoff(rows, fig_dir)
        _fig_tau(rows, fig_dir)

    # semisup 专属
    if mode == "semisup":
        _fig_label_efficiency(rows, fig_dir)
        _fig_alpha_comparison(rows, fig_dir)

    _latex_table(rows, save_dir, mode)
    _text_table(rows, save_dir, mode)
    return rows


def _build_rows(grouped, ordered):
    rows = []
    for name in ordered:
        runs = grouped[name]
        def _ms(key): return np.mean([r[key] for r in runs]), np.std([r[key] for r in runs])
        am, astd = _ms("post_acc");   nm, ns = _ms("nmi")
        rm, rs   = _ms("recon_loss"); xm, xs = _ms("xcond")
        acm, acs = _ms("n_active");   pm, ps = _ms("pi_entropy")
        rows.append(dict(
            name=name, display=runs[0]["display_name"],
            acc_mean=am, acc_std=astd, nmi_mean=nm, nmi_std=ns,
            recon_mean=rm, recon_std=rs, xcond_mean=xm, xcond_std=xs,
            active_mean=acm, active_std=acs, pi_ent_mean=pm, pi_ent_std=ps,
            K_model=runs[0].get("K_model", 10),
            latent_dim=runs[0].get("latent_dim", 2),
            beta=runs[0].get("beta", 2.0),
            alpha_unlabeled=runs[0].get("alpha_unlabeled", 0.5),
            labeled_per_class=runs[0].get("labeled_per_class", 100),
            n_runs=len(runs),
        ))
    return rows


# ---------- 图 A1: 主指标柱状 ----------
def _fig_main(rows, fig_dir, mode):
    baseline = "full_model" if mode == "unsup" else "ss_full"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(rows)); w = 0.6

    specs = [
        ("acc_mean", "acc_std", "Accuracy ↑", "(a) Clustering Accuracy",
         (0, 1.05), lambda v: f'{v:.1%}'),
        ("recon_mean", "recon_std", "Recon Loss ↓", "(b) Reconstruction",
         None, lambda v: f'{v:.1f}'),
        ("xcond_mean", "xcond_std", "x-Cond ↑", "(c) Decoder x-Dependence",
         (0, 1.05), lambda v: f'{v:.2f}'),
    ]
    for i, (mk, sk, yl, tt, ylim, fmt) in enumerate(specs):
        ax = axes[i]
        vals = [r[mk] for r in rows]
        errs = [r[sk] for r in rows]
        cols = [STYLE["primary"] if r["name"] == baseline else STYLE["neutral"]
                for r in rows]
        bars = ax.bar(x, vals, w, yerr=errs, capsize=3,
                      color=cols, edgecolor='white', lw=0.8)
        ax.set_ylabel(yl); ax.set_title(tt, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r["display"] for r in rows],
                           rotation=40, ha='right', fontsize=8)
        if ylim: ax.set_ylim(*ylim)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+e+0.01,
                    fmt(v), ha='center', va='bottom', fontsize=7)

    fig.suptitle(f"mVAE Ablation ({mode})", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"ablation_main_{mode}.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A2: Active + Pi Entropy ----------
def _fig_active_pi(rows, fig_dir, mode):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(rows)); w = 0.6; labels = [r["display"] for r in rows]

    ax = axes[0]
    ax.bar(x, [r["active_mean"] for r in rows], w,
           yerr=[r["active_std"] for r in rows], capsize=3,
           color='#4CAF50', edgecolor='white')
    for i, r in enumerate(rows):
        ax.plot([i-0.3, i+0.3], [r["K_model"]]*2, 'r--', lw=1, alpha=0.5)
    ax.set_ylabel("# Active ($\\pi_k>0.02$)")
    ax.set_title("(a) Active Classes", fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)

    ax = axes[1]
    ax.bar(x, [r["pi_ent_mean"] for r in rows], w,
           yerr=[r["pi_ent_std"] for r in rows], capsize=3,
           color='#FF9800', edgecolor='white')
    for i, r in enumerate(rows):
        ax.plot([i-0.3, i+0.3], [np.log(r["K_model"])]*2, 'g--', lw=1, alpha=0.5)
    ax.set_ylabel("$\\Pi$ Entropy")
    ax.set_title("(b) Class Prior Entropy (green=uniform)", fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"ablation_active_pi_{mode}.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A3: 正则化消融 (unsup) ----------
def _fig_reg_ablation(rows, fig_dir):
    sub = [r for r in rows if r["name"] in
           ["full_model", "no_z_dropout", "no_balance", "no_both_reg"]]
    if len(sub) < 2: return
    cmap = {"full_model": "#2196F3", "no_z_dropout": "#FF9800",
            "no_balance": "#E91E63", "no_both_reg": "#9E9E9E"}

    specs = [("acc_mean", "acc_std", "Post. Accuracy ↑"),
             ("xcond_mean", "xcond_std", "x-Conditionality ↑"),
             ("recon_mean", "recon_std", "Recon Loss ↓"),
             ("pi_ent_mean", "pi_ent_std", "$\\Pi$ Entropy ↑")]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(sub))
    for i, (mk, sk, tl) in enumerate(specs):
        ax = axes[i]
        ax.bar(x, [r[mk] for r in sub], 0.6,
               yerr=[r[sk] for r in sub], capsize=4,
               color=[cmap.get(r["name"], "#9E9E9E") for r in sub],
               edgecolor='white')
        ax.set_title(tl, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r["display"] for r in sub], rotation=30,
                           ha='right', fontsize=8)
    fig.suptitle("Regularization Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_regularization.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A4: dim×beta 热力图 (unsup) ----------
def _fig_dim_beta_heatmap(rows, fig_dir):
    data = {(r["latent_dim"], r["beta"]): r for r in rows}
    dims  = sorted(set(k[0] for k in data))
    betas = sorted(set(k[1] for k in data))
    if len(dims) < 2 or len(betas) < 2: return

    for mk, label, cm in [("acc_mean", "Accuracy", "YlOrRd"),
                           ("xcond_mean", "x-Cond", "YlGnBu")]:
        mat = np.full((len(betas), len(dims)), np.nan)
        for bi, b in enumerate(betas):
            for di, d in enumerate(dims):
                if (d, b) in data: mat[bi, di] = data[(d, b)][mk]
        if np.isnan(mat).all(): continue

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, cmap=cm, aspect='auto')
        ax.set_xticks(range(len(dims)));  ax.set_xticklabels(dims)
        ax.set_yticks(range(len(betas))); ax.set_yticklabels(betas)
        ax.set_xlabel("$d_z$"); ax.set_ylabel("$\\beta$")
        ax.set_title(f"{label} ($d_z \\times \\beta$)", fontweight='bold')
        for bi in range(len(betas)):
            for di in range(len(dims)):
                if not np.isnan(mat[bi, di]):
                    fmt = f'{mat[bi,di]:.1%}' if 'acc' in mk else f'{mat[bi,di]:.3f}'
                    c = 'white' if mat[bi,di] > np.nanmedian(mat) else 'black'
                    ax.text(di, bi, fmt, ha='center', va='center',
                            fontsize=10, fontweight='bold', color=c)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"ablation_heatmap_{mk.replace('_mean','')}.png"),
                    dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A5: dim trade-off (unsup) ----------
def _fig_dim_tradeoff(rows, fig_dir):
    sub = sorted([r for r in rows if r["name"] in
                  ["full_model", "zdim_4", "zdim_8", "zdim_16"]],
                 key=lambda r: r["latent_dim"])
    if len(sub) < 2: return
    dims = [r["latent_dim"] for r in sub]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.errorbar(dims, [r["acc_mean"] for r in sub],
                 yerr=[r["acc_std"] for r in sub],
                 marker='o', color='#2196F3', lw=2, capsize=5, label='Accuracy ↑')
    ax1.errorbar(dims, [r["xcond_mean"] for r in sub],
                 yerr=[r["xcond_std"] for r in sub],
                 marker='D', color='#4CAF50', lw=2, capsize=5, label='x-Cond ↑')
    ax1.set_xlabel("$d_z$"); ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.05)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(dims); ax1.set_xticklabels(dims)

    ax2 = ax1.twinx()
    ax2.errorbar(dims, [r["recon_mean"] for r in sub],
                 yerr=[r["recon_std"] for r in sub],
                 marker='s', color='#E91E63', lw=2, capsize=5, ls='--', label='Recon ↓')
    ax2.set_ylabel("Recon Loss", color='#E91E63')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right')
    ax1.set_title("$d_z$ Trade-off", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_dim_tradeoff.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A6: 收敛曲线 (通用) ----------
def _fig_convergence(grouped, ordered, save_dir, fig_dir, mode):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cm = plt.cm.tab20
    for i, name in enumerate(ordered):
        hp = os.path.join(save_dir, f"{name}_seed{SEEDS[0]}", "result.json")
        if not os.path.exists(hp): continue
        with open(hp) as f: h = json.load(f).get("history", [])
        if not h: continue
        c = cm(i / max(len(ordered), 1))
        dn = grouped[name][0]["display_name"]
        axes[0].plot([e["epoch"] for e in h], [e.get("post_acc", 0) for e in h],
                     lw=1.2, alpha=0.8, color=c, label=dn)
        axes[1].plot([e["epoch"] for e in h], [e.get("recon", 0) for e in h],
                     lw=1.2, alpha=0.8, color=c, label=dn)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Post. Accuracy")
    axes[0].set_title("(a) Accuracy", fontweight='bold')
    axes[0].legend(fontsize=7, ncol=2); axes[0].set_ylim(0, 1)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recon Loss")
    axes[1].set_title("(b) Reconstruction", fontweight='bold')
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Convergence ({mode})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"ablation_convergence_{mode}.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A7: τ 策略 (unsup) ----------
def _fig_tau(rows, fig_dir):
    sub = [r for r in rows if r["name"] in
           ["full_model", "tau_aggressive", "tau_conservative"]]
    if len(sub) < 2: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(sub))
    cols = ['#2196F3', '#E91E63', '#FF9800']
    for i, (mk, yl, tt) in enumerate([
        ("acc_mean", "Accuracy", "(a)"),
        ("xcond_mean", "x-Cond", "(b)"),
        ("pi_ent_mean", "$\\Pi$ Ent", "(c)"),
    ]):
        ax = axes[i]
        ax.bar(x, [r[mk] for r in sub], 0.5,
               yerr=[r[mk.replace("mean","std")] for r in sub], capsize=4,
               color=cols[:len(sub)], edgecolor='white')
        ax.set_ylabel(yl); ax.set_title(tt, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r["display"] for r in sub], fontsize=9)
    fig.suptitle("Gumbel $\\tau$ Strategy", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_tau.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A8: Label Efficiency (semisup) ----------
def _fig_label_efficiency(rows, fig_dir):
    sub = sorted([r for r in rows if r["name"] in
                  ["ss_label_10", "ss_label_50", "ss_full", "ss_label_500"]],
                 key=lambda r: r["labeled_per_class"])
    if len(sub) < 2: return
    lpc = [r["labeled_per_class"] for r in sub]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.errorbar(lpc, [r["acc_mean"] for r in sub],
                 yerr=[r["acc_std"] for r in sub],
                 marker='o', color='#2196F3', lw=2, capsize=5, label='Accuracy')
    ax1.set_xlabel("Labels per class"); ax1.set_ylabel("Accuracy")
    ax1.set_xscale('log'); ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.errorbar(lpc, [r["recon_mean"] for r in sub],
                 yerr=[r["recon_std"] for r in sub],
                 marker='s', color='#E91E63', lw=2, capsize=5, ls='--', label='Recon')
    ax2.set_ylabel("Recon Loss", color='#E91E63')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    ax1.set_title("Label Efficiency", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_label_efficiency.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- 图 A9: α_unlabeled (semisup) ----------
def _fig_alpha_comparison(rows, fig_dir):
    sub = [r for r in rows if r["name"] in
           ["ss_full", "ss_alpha_0.1", "ss_alpha_1.0"]]
    if len(sub) < 2: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(sub)); cols = ['#2196F3', '#FF9800', '#4CAF50']
    labels = [r["display"] for r in sub]

    for i, (mk, yl, tt) in enumerate([
        ("acc_mean", "Accuracy", "(a) Accuracy vs $\\alpha_{un}$"),
        ("recon_mean", "Recon Loss", "(b) Recon vs $\\alpha_{un}$"),
    ]):
        ax = axes[i]
        ax.bar(x, [r[mk] for r in sub], 0.5,
               yerr=[r[mk.replace("mean","std")] for r in sub], capsize=4,
               color=cols[:len(sub)], edgecolor='white')
        ax.set_ylabel(yl); ax.set_title(tt, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    fig.suptitle("$\\alpha_{unlabeled}$ Sensitivity", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_alpha.png"),
                dpi=200, bbox_inches='tight'); plt.close()


# ---------- LaTeX 表格 ----------
def _latex_table(rows, save_dir, mode):
    baseline = "full_model" if mode == "unsup" else "ss_full"
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{mVAE ablation (" + mode + r").}",
        r"\label{tab:mvae_" + mode + r"}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Config & Acc $\uparrow$ & NMI $\uparrow$ & Recon $\downarrow$ "
        r"& x-Cond $\uparrow$ & Active \\", r"\midrule",
    ]
    for r in rows:
        nm = r["display"].replace("_", r"\_")
        if r["name"] == baseline: nm = r"\textbf{" + nm + "}"
        a = f"${r['acc_mean']:.1%}\\pm{r['acc_std']:.1%}$".replace("%",r"\%")
        n = f"${r['nmi_mean']:.4f}\\pm{r['nmi_std']:.4f}$"
        rc= f"${r['recon_mean']:.1f}\\pm{r['recon_std']:.1f}$"
        xc= f"${r['xcond_mean']:.3f}\\pm{r['xcond_std']:.3f}$"
        ac= f"${r['active_mean']:.1f}/{r['K_model']}$"
        lines.append(f"  {nm} & {a} & {n} & {rc} & {xc} & {ac} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    p = os.path.join(save_dir, f"ablation_table_{mode}.tex")
    with open(p, 'w') as f: f.write("\n".join(lines))
    print(f"  LaTeX → {p}")


def _text_table(rows, save_dir, mode):
    p = os.path.join(save_dir, f"ablation_table_{mode}.txt")
    with open(p, 'w') as f:
        hdr = f"{'Config':<28s} {'Acc':>16s} {'NMI':>14s} {'Recon':>14s} {'xCond':>14s} {'Act':>8s}"
        f.write(hdr + "\n" + "─"*len(hdr) + "\n")
        for r in rows:
            f.write(
                f"{r['display']:<28s} "
                f"{r['acc_mean']:.1%}±{r['acc_std']:.1%}   "
                f"{r['nmi_mean']:.4f}±{r['nmi_std']:.4f}   "
                f"{r['recon_mean']:.1f}±{r['recon_std']:.1f}   "
                f"{r['xcond_mean']:.3f}±{r['xcond_std']:.3f}   "
                f"{r['active_mean']:.1f}/{r['K_model']}\n")
    print(f"\n{'='*70}\nmVAE ABLATION — {mode.upper()}\n{'='*70}")
    with open(p) as f: print(f.read())


# ==============================================================
# CLI
# ==============================================================
if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="mVAE Ablation (Phase 0 + Phase 1)")
    pa.add_argument("--mode", default="unsup", choices=["unsup","semisup","both"])
    pa.add_argument("--plot-only", action="store_true",
                    help="Phase 1: 仅从已有结果出图")
    pa.add_argument("--config", type=str, default=None,
                    help="Phase 1: 只跑指定配置")
    pa.add_argument("--seed", type=int, default=None,
                    help="Phase 1: 只跑指定种子")
    pa.add_argument("--save-dir", type=str, default=None)
    pa.add_argument("--quick", action="store_true",
                    help="Phase 1: 30 epochs 快速验证")
    pa.add_argument("--optuna", action="store_true",
                    help="Phase 0: 运行 Optuna 搜索最优 baseline")
    pa.add_argument("--n-trials", type=int, default=100,
                    help="Phase 0: Optuna trial 数 (default: 100)")
    args = pa.parse_args()

    # ============================================================
    # Phase 0: Optuna baseline 搜索
    # ============================================================
    if args.optuna:
        modes = ["unsup", "semisup"] if args.mode == "both" else [args.mode]
        for m in modes:
            sd = args.save_dir or f"mVAE_ablation_{m}"
            run_optuna_search(mode=m, n_trials=args.n_trials, save_dir=sd)
        print("\n✓ Phase 0 完成. 接下来运行消融:")
        print(f"  python mVAE_ablation.py --mode {args.mode}")
        sys.exit(0)

    # ============================================================
    # Phase 1: 消融实验
    # ============================================================
    if args.quick:
        quick_ep = 30
    else:
        quick_ep = None

    if args.plot_only:
        for m in (["unsup","semisup"] if args.mode == "both" else [args.mode]):
            sd = args.save_dir or f"mVAE_ablation_{m}"
            r = load_results(sd)
            if r: generate_ablation_report(r, sd, m)
            else: print(f"[{m}] No results in {sd}")
    else:
        cfgs = [args.config] if args.config else None
        sds  = [args.seed]   if args.seed  else None
        results = run_all_experiments(
            mode=args.mode, configs_to_run=cfgs,
            seeds=sds, save_dir=args.save_dir,
            quick_epochs=quick_ep)
        for m in (["unsup","semisup"] if args.mode == "both" else [args.mode]):
            sd = args.save_dir or f"mVAE_ablation_{m}"
            mr = [r for r in results if r.get("mode") == m]
            if mr: generate_ablation_report(mr, sd, m)