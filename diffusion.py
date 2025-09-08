import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from config import T

# =====================
# Beta schedule
# =====================
def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)  # 线性调度

betas = make_beta_schedule(T)  # (T,)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=-1)  # \bar{alpha}_t
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)  # \bar{alpha}_{t-1}

# 方差项，用在反向采样里（无监督时近似后验也要用）
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

# ==============
# 工具函数
# ==============
def extract(a, t, x_shape):
    """从时间序列 a 中取出 batch 对应的值，并 reshape 成 [B,1,1,1]"""
    a = a.to(t.device)
    idx = (t - 1).clamp(min=0)  # t=1 → idx=0
    out = a.gather(0, idx)
    return out.view((t.size(0),) + (1,) * (len(x_shape) - 1))

# =====================
# 标准前向扩散 (Supervised/Uncond)
# =====================
def forward_add_noise(x0, t):
    """
    给定原始图像 x0 和时间步 t，生成加噪后的 x_t 和对应噪声 epsilon
    """
    noise = torch.randn_like(x0)
    a_bar_t = extract(alphas_cumprod, t, x0.shape)
    x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * noise
    return x_t, noise

# =====================
# 无监督 mixture diffusion 版本
# =====================
@torch.no_grad()
def forward_add_noise_pair(x0, t):
    """
    给定原始图像 x0 和时间步 t (t >= 1)，生成一对 (z_{t-1}, z_t)，以及等效噪声 eps_eff
    用于计算类别后验 p(x|z_{t-1}, z_t)
    
    这个版本严格按照文档，区分了 t=1 和 t>1 的情况
    """
    B = x0.size(0)
    # 将噪声张量移动到输入数据所在设备
    eps_star = torch.randn_like(x0, device=x0.device) # 用于 z_t
    
    # 区分 t=1 和 t>1 的情况
    is_t1_mask = (t == 1)
    
    # ------------------- 处理 t > 1 的情况 -------------------
    # 在非 t=1 的样本上操作
    t_greater_1 = t[~is_t1_mask]
    x0_greater_1 = x0[~is_t1_mask]
    eps_star_greater_1 = eps_star[~is_t1_mask]
    
    # 采样 eps 并确保在同一设备
    eps_greater_1 = torch.randn_like(x0_greater_1, device=x0.device)
    
    # 提取相应的参数 - 显式指定设备为x0.device
    a_bar_tm1_greater_1 = extract(alphas_cumprod_prev.to(x0.device), t_greater_1, x0_greater_1.shape)
    a_t_greater_1 = extract(alphas.to(x0.device), t_greater_1, x0_greater_1.shape)
    
    # 计算 z_{t-1} 和 z_t
    z_tm1_greater_1 = torch.sqrt(a_bar_tm1_greater_1) * x0_greater_1 + torch.sqrt(1.0 - a_bar_tm1_greater_1) * eps_greater_1
    z_t_greater_1 = torch.sqrt(a_t_greater_1) * z_tm1_greater_1 + torch.sqrt(1.0 - a_t_greater_1) * eps_star_greater_1

    # 计算等效噪声 eps_eff
    a_bar_t_greater_1 = extract(alphas_cumprod.to(x0.device), t_greater_1, x0_greater_1.shape)
    one_m_a_bar_t_greater_1 = 1.0 - a_bar_t_greater_1
    
    eps_eff_greater_1 = ( torch.sqrt(a_t_greater_1) * torch.sqrt(1.0 - a_bar_tm1_greater_1) * eps_greater_1
                        + torch.sqrt(1.0 - a_t_greater_1) * eps_star_greater_1 ) / torch.sqrt(one_m_a_bar_t_greater_1)

    # ------------------- 处理 t = 1 的情况 -------------------
    # 确保零张量在正确设备上
    z_tm1_t1 = torch.zeros_like(x0[is_t1_mask], device=x0.device) # z_{t-1} 在 t=1 时不存在，可以设为零或不使用
    
    a_t_t1 = extract(alphas.to(x0.device), t[is_t1_mask], x0[is_t1_mask].shape)
    
    # 文档中的 t=1 公式
    z_t_t1 = torch.sqrt(a_t_t1) * x0[is_t1_mask] + torch.sqrt(1.0 - a_t_t1) * eps_star[is_t1_mask]
    
    # 文档中 t=1 的等效噪声
    eps_eff_t1 = eps_star[is_t1_mask]

    # ------------------- 拼接结果 -------------------
    # 创建完整的张量并确保在正确设备上
    z_t = torch.zeros_like(x0, device=x0.device)
    z_tm1 = torch.zeros_like(x0, device=x0.device)
    eps_eff = torch.zeros_like(x0, device=x0.device)
    
    # 填充结果
    z_t[~is_t1_mask] = z_t_greater_1
    z_tm1[~is_t1_mask] = z_tm1_greater_1
    eps_eff[~is_t1_mask] = eps_eff_greater_1
    
    z_t[is_t1_mask] = z_t_t1
    z_tm1[is_t1_mask] = z_tm1_t1
    eps_eff[is_t1_mask] = eps_eff_t1

    return z_t, z_tm1, eps_eff

# ======================
# 类别后验计算
# ======================
def posterior_logit(z_t, z_tm1, t, eps_pred):
    """
    计算 log ξ_k ≈ -|| z_{t-1} - μ_theta ||^2 / (2 * beta_tilde_t)
    """
    a_t = extract(alphas, t, z_t.shape)
    b_t = extract(betas, t, z_t.shape)
    a_bar_t = extract(alphas_cumprod, t, z_t.shape)
    a_bar_tm1 = extract(alphas_cumprod_prev, t, z_t.shape)

    mu = (1.0 / torch.sqrt(a_t)) * (z_t - (b_t / torch.sqrt(1.0 - a_bar_t)) * eps_pred)
    beta_tilde = (1.0 - a_bar_tm1) / (1.0 - a_bar_t) * b_t

    diff = z_tm1 - mu
    logits = - (diff.pow(2).sum(dim=(1,2,3))) / (2.0 * beta_tilde.view(-1))
    return logits

@torch.no_grad()
def posterior_probs(model, z_t, z_tm1, t, K, eps_eff, tau=1.0, log_prior=None, use_decoder_at_t1=True):
    # ---------- 情况 1: t > 1 ----------
    mask = (t > 1)
    B = z_t.size(0)
    device = z_t.device

    logits_full = torch.zeros(B, K, device=device)

    if mask.any():
        z_t_m   = z_t[mask]
        z_tm1_m = z_tm1[mask]
        t_m     = t[mask]

        logits_list = []
        for k in range(K):
            yk = torch.full((z_t_m.size(0),), k, dtype=torch.long, device=device)
            eps_pred = model(z_t_m, t_m, yk)
            logit_k = posterior_logit(z_t_m, z_tm1_m, t_m, eps_pred)  # [Bm]
            logits_list.append(logit_k)

        logits_m = torch.stack(logits_list, dim=1)  # [Bm, K]
        if log_prior is not None:
            logits_m = logits_m + log_prior.view(1, K)  # 加上先验的 log π_k（若非均匀）

        logits_full[mask] = logits_m

    # ---------- 情况 2: t = 1 ----------
    mask_t1 = (t == 1)
    if use_decoder_at_t1 and mask_t1.any():
        z_t1   = z_t[mask_t1]
        eps_t1 = eps_eff[mask_t1]  # 这一步的等效噪声

        logits_list = []
        for k in range(K):
            yk = torch.full((z_t1.size(0),), k, dtype=torch.long, device=device)
            eps_pred = model(z_t1, t[mask_t1], yk)
            # 直接用 -MSE 作为 decoder likelihood 的 proxy
            mse = F.mse_loss(eps_pred, eps_t1, reduction="none").sum(dim=(1,2,3))
            logits_k = -mse
            logits_list.append(logits_k)

        logits_full[mask_t1] = torch.stack(logits_list, dim=1)

    # ---------- Softmax ----------
    probs = F.softmax(logits_full / tau, dim=1)
    return probs



if __name__=='__main__':
    import matplotlib.pyplot as plt 
    from mnist_data import MNIST
    
    dataset=MNIST()
    x=torch.stack((dataset[0][0],dataset[1][0]),dim=0) # 2个图片拼batch, (2,1,48,48)

    # 原图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(x[0].permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(x[1].permute(1,2,0))
    plt.show()

    # 随机时间步
    t=torch.randint(0,T,size=(x.size(0),))
    print('t:',t)

    # 加噪
    x=x*2-1 # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    x,noise=forward_add_noise(x,t)

    print('x:',x.size())
    print('noise:',noise.size())

    # 加噪图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(((x[0]+1)/2).permute(1,2,0))   
    plt.subplot(1,2,2)
    plt.imshow(((x[1]+1)/2).permute(1,2,0))
    plt.show()

    z_t, z_tm1, eps_eff = forward_add_noise_pair(x, t)
    print('z_t:',z_t.size())
    print('z_tm1:',z_tm1.size())
    print('eps_eff:',eps_eff.size())

    # 加噪图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(((z_t[0]+1)/2).permute(1,2,0))   
    plt.subplot(1,2,2)
    plt.imshow(((z_t[1]+1)/2).permute(1,2,0))
    plt.show()

    # 加噪图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(((z_tm1[0]+1)/2).permute(1,2,0))   
    plt.subplot(1,2,2)
    plt.imshow(((z_tm1[1]+1)/2).permute(1,2,0))
    plt.show()