import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from VAE_config import *
from torchvision.utils import save_image
from scipy.stats import mode
import os

# --- 4. EM 算法实现 ---
# 创建一个新的 e_step 函数或修改原来的
def e_step(model, dataloader, label_priors, num_classes, device, beta=1.0):
    """
    E-Step: 使用完整的、带beta权重的ELBO来计算后验概率。
    """
    model.eval()
    all_gamma = []
    all_true_labels = []
    
    print("Performing E-Step (using full weighted ELBO)...")
    with torch.no_grad():
        for y_batch, true_labels in tqdm(dataloader):
            y_batch = y_batch.to(device)
            batch_size = y_batch.size(0)

            y_tiled = y_batch.unsqueeze(1).repeat(1, num_classes, 1, 1, 1).view(-1, 1, 28, 28)
            x_onehot = torch.eye(num_classes, device=device)
            x_onehot_tiled = x_onehot.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, num_classes)

            # 通过模型获得所有参数
            recon_tiled, mu_tiled, logvar_tiled = model(y_tiled, x_onehot_tiled)
            
            # --- 核心修改：使用完整的加权ELBO ---
            recon_loss_part = F.binary_cross_entropy(recon_tiled, y_tiled, reduction='none').sum(dim=[1, 2, 3])
            kld_part = -0.5 * torch.sum(1 + logvar_tiled - mu_tiled.pow(2) - logvar_tiled.exp(), dim=1)
            
            # 计算负ELBO，注意KL项要乘以当前的beta
            neg_elbo_flat = recon_loss_part + beta * kld_part
            neg_elbo_batched = neg_elbo_flat.view(batch_size, num_classes)
            
            # 对数似然 log p(y|x) 就用 -neg_elbo 来近似
            log_likelihood = -neg_elbo_batched
            # --- 修改结束 ---

            log_p_y_x = log_likelihood + torch.log(label_priors).unsqueeze(0)
            log_p_y = torch.logsumexp(log_p_y_x, dim=1, keepdim=True)
            log_gamma = log_p_y_x - log_p_y
            gamma_batch = torch.exp(log_gamma)
            all_gamma.append(gamma_batch.cpu())
            all_true_labels.append(true_labels)

    return torch.cat(all_gamma, dim=0), torch.cat(all_true_labels, dim=0)

def m_step(model, optimizer, dataloader, gamma, beta=1.0):
    """
    M-Step: 更新模型参数和类别先验
    """
    model.train()
    # 创建一个新的 DataLoader，包含数据和对应的 gamma
    dataset_with_gamma = TensorDataset(dataloader.tensors[0], gamma)
    loader_with_gamma = DataLoader(dataset_with_gamma, batch_size=BATCH_SIZE, shuffle=True)

    print("Performing M-Step...")
    for _ in range(NUM_M_STEP_EPOCHS):
        epoch_loss = 0
        for y_batch, gamma_batch in tqdm(loader_with_gamma):
            y_batch, gamma_batch = y_batch.to(DEVICE), gamma_batch.to(DEVICE)
            optimizer.zero_grad()
            # 同样地，准备输入以计算所有类别的损失    
            y_tiled = y_batch.unsqueeze(1).repeat(1, NUM_CLASSES, 1, 1, 1).view(-1, 1, 28, 28)
            x_onehot = torch.eye(NUM_CLASSES, device=DEVICE)
            x_onehot_tiled = x_onehot.unsqueeze(0).repeat(y_batch.size(0), 1, 1).view(-1, NUM_CLASSES)
            
            recon_flat, mu_flat, logvar_flat = model(y_tiled, x_onehot_tiled)

            # 分别计算重构损失和KL散度
            recon_loss_part = F.binary_cross_entropy(recon_flat, y_tiled, reduction='none').sum(dim=[1, 2, 3])
            kld_part = -0.5 * torch.sum(1 + logvar_flat - mu_flat.pow(2) - logvar_flat.exp(), dim=1)

            # 使用 beta 加权 KL 散度
            neg_elbo_flat = recon_loss_part + beta * kld_part
            # 计算负ELBO (即损失)
            # neg_elbo_flat = loss_function(recon_flat, y_tiled, mu_flat, logvar_flat)
            neg_elbo_batched = neg_elbo_flat.view(y_batch.size(0), NUM_CLASSES)

            # 计算加权损失
            weighted_loss = torch.sum(gamma_batch * neg_elbo_batched)
            
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()
            
        print(f"  M-Step Epoch Loss: {epoch_loss / len(dataloader):.4f}")

    # 更新类别先验 p(x)
    new_label_priors = torch.mean(gamma, dim=0)
    
    return new_label_priors, epoch_loss / len(dataloader)

def m_step_sampled(model, optimizer, dataloader, gamma, beta=1.0):
    """
    M-Step (Sampled EM): 从 gamma 分布中采样类别作为伪标签，
    并统计类别计数来更新先验。
    """
    model.train()

    # 统计每个类别的计数
    class_counts = torch.zeros(NUM_CLASSES, device=DEVICE)

    # === 先采样标签 ===
    # sampled_labels = torch.multinomial(gamma, num_samples=1).squeeze(1).to(DEVICE)   # (N,)
    # 选择最大概率的类别作为伪标签
    sampled_labels = torch.argmax(gamma, dim=1).to(DEVICE)   # (N,)
    # 直接做 bincount 来统计类别分布
    class_counts += torch.bincount(sampled_labels, minlength=NUM_CLASSES).float()
    new_label_priors = class_counts / class_counts.sum()  # 归一化

    # === 构建伪标签 one-hot ===
    sampled_onehot = F.one_hot(sampled_labels, NUM_CLASSES).float()
    dataset_with_labels = TensorDataset(dataloader.tensors[0], sampled_onehot)
    loader_with_labels = DataLoader(dataset_with_labels, batch_size=BATCH_SIZE, shuffle=True)

    print("Performing Sampled M-Step...")
    total_loss = 0
    num_batches = 0

    for _ in range(NUM_M_STEP_EPOCHS):
        epoch_loss = 0
        for y_batch, sampled_onehot in tqdm(loader_with_labels):
            y_batch, sampled_onehot = y_batch.to(DEVICE), sampled_onehot.to(DEVICE)  # (B, 1, 28, 28), (B, K)
            optimizer.zero_grad()

            # 前向传播
            recon, mu, logvar = model(y_batch, sampled_onehot)

            # VAE 损失
            recon_loss_part = F.binary_cross_entropy(recon, y_batch, reduction='none').sum(dim=[1, 2, 3])
            kld_part = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            # 使用 beta 加权 KL 散度
            loss = (recon_loss_part + beta * kld_part).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1

        print(f"  Sampled M-Step Epoch Loss: {epoch_loss / len(loader_with_labels):.4f}")

    avg_loss = total_loss / num_batches
    return new_label_priors, avg_loss


# -----------------
# 1. 在脚本开头定义超参数
# -----------------
# 对比损失的权重。这是一个关键的超参数，需要调试。
# 建议从 0.1, 0.5, 1.0 开始尝试。
lambda_contrast = 0.5

def m_step_contrastive(model, optimizer, dataloader, gamma, beta=1.0):
    """
    M-Step with Contrastive Learning for VAE.
    在潜空间的均值向量 mu 上施加斥力。
    """
    model.train()

    # --- 这部分与您之前的逻辑完全相同：采样标签、计算新先验、创建DataLoader ---
    class_counts = torch.zeros(NUM_CLASSES, device=DEVICE)
    # sampled_labels = torch.multinomial(gamma, num_samples=1).squeeze(1).to(DEVICE)
    # 选择最大概率的类别作为伪标签
    sampled_labels = torch.argmax(gamma, dim=1).to(DEVICE)   # (N,)
    class_counts += torch.bincount(sampled_labels, minlength=NUM_CLASSES).float()
    new_label_priors = class_counts / class_counts.sum()

    sampled_onehot = F.one_hot(sampled_labels, NUM_CLASSES).float()
    dataset_with_labels = TensorDataset(dataloader.tensors[0], sampled_onehot)
    loader_with_labels = DataLoader(dataset_with_labels, batch_size=BATCH_SIZE, shuffle=True)
    # --- 采样逻辑结束 ---

    print("Performing Contrastive M-Step for VAE...")
    total_loss = 0
    num_batches = 0

    for _ in range(NUM_M_STEP_EPOCHS):
        epoch_loss = 0
        for y_batch, x_correct_onehot in tqdm(loader_with_labels):
            y_batch, x_correct_onehot = y_batch.to(DEVICE), x_correct_onehot.to(DEVICE)
            batch_size = y_batch.size(0)
            optimizer.zero_grad()
            
            # ======================= 对比学习核心修改 =======================
            
            # 1. 生成“错误”标签 x_wrong
            x_correct_indices = torch.argmax(x_correct_onehot, dim=1)
            rand_offset = torch.randint(1, NUM_CLASSES, (batch_size,), device=DEVICE)
            x_wrong_indices = (x_correct_indices + rand_offset) % NUM_CLASSES
            x_wrong_onehot = F.one_hot(x_wrong_indices, NUM_CLASSES).float()

            # 2. 高效地进行两次编码
            # 我们将 correct 和 wrong 的输入拼接，只调用一次编码器
            y_combined = torch.cat([y_batch, y_batch], dim=0)
            x_combined = torch.cat([x_correct_onehot, x_wrong_onehot], dim=0)
            
            # 编码器输出两倍批次大小的 mu 和 logvar
            mu_combined, logvar_combined = model.encode(y_combined, x_combined)
            
            # 拆分回 correct 和 wrong 两部分
            mu_correct, mu_wrong = torch.chunk(mu_combined, 2, dim=0)
            logvar_correct, _ = torch.chunk(logvar_combined, 2, dim=0) # logvar_wrong 不需要

            # 3. 计算“引力”部分：标准的 VAE 损失
            # 这部分需要完整的重构过程
            z_correct = model.reparameterize(mu_correct, logvar_correct)
            recon_correct = model.decode(z_correct, x_correct_onehot)
            
            recon_loss = F.binary_cross_entropy(recon_correct, y_batch, reduction='none').sum(dim=[1, 2, 3])
            kld_loss = -0.5 * torch.sum(1 + logvar_correct - mu_correct.pow(2) - logvar_correct.exp(), dim=1)
            
            standard_loss = (recon_loss + beta * kld_loss).mean()

            # 4. 计算“斥力”部分：对比损失
            # 我们希望 mu_correct 和 mu_wrong 尽可能远
            # 最大化 ||mu_correct - mu_wrong||^2 等价于最小化 -||mu_correct - mu_wrong||^2
            contrastive_loss = -F.mse_loss(mu_correct, mu_wrong)

            # 5. 合并总损失
            total_loss_batch = standard_loss + lambda_contrast * contrastive_loss
            
            total_loss_batch.backward()
            # 防止梯度爆炸
            # Clips the gradients to a maximum norm of 1.0. This is a common value.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # --------------------
            optimizer.step()

            epoch_loss += total_loss_batch.item()
            total_loss += total_loss_batch.item()
            num_batches += 1
            
        print(f"  Contrastive M-Step Epoch Loss: {epoch_loss / len(loader_with_labels):.4f}")
        
    avg_loss = total_loss / num_batches  # 修正avg_loss的计算
    return new_label_priors, avg_loss

# =================================================================
#  == 核心修改：带有自洽性约束的M-Step ==
# =================================================================
def m_step_with_classifier(
    generator, classifier, optimizer_g, optimizer_c,
    dataloader, gamma, beta=1.0, lambda_consistency=1.0
):
    """
    M-Step with a self-consistency classifier loss.
    
    Args:
        generator: The VAE model.
        classifier: The auxiliary classifier model.
        optimizer_g: Optimizer for the generator (VAE).
        optimizer_c: Optimizer for the classifier.
        dataloader: Provides the raw image data.
        gamma: The soft labels from the E-step.
        beta: Weight for the KL divergence term.
        lambda_consistency: Weight for the new classifier consistency loss.
    """
    generator.train()
    classifier.train()

    # --- 采样伪标签 (与您之前的逻辑相同) ---
    sampled_labels = torch.multinomial(gamma, num_samples=1).squeeze(1).to(DEVICE)
    class_counts = torch.bincount(sampled_labels, minlength=NUM_CLASSES).float()
    new_label_priors = class_counts / class_counts.sum()
    sampled_onehot = F.one_hot(sampled_labels, NUM_CLASSES).float()
    
    dataset_with_labels = TensorDataset(dataloader.tensors[0], sampled_onehot, sampled_labels)
    loader_with_labels = DataLoader(dataset_with_labels, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Performing M-Step with Classifier Self-Consistency...")
    total_g_loss = 0
    total_c_loss = 0
    num_batches = 0

    for _ in range(NUM_M_STEP_EPOCHS):
        for y_batch, x_onehot, x_indices in tqdm(loader_with_labels):
            y_batch, x_onehot, x_indices = y_batch.to(DEVICE), x_onehot.to(DEVICE), x_indices.to(DEVICE)
            
            # --- 步骤 1: 训练生成器 (VAE) ---
            optimizer_g.zero_grad()
            
            # 1a. 标准VAE前向传播和损失
            recon, mu, logvar = generator(y_batch, x_onehot)
            recon_loss = F.binary_cross_entropy(recon, y_batch, reduction='none').sum(dim=[1, 2, 3])
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            vae_loss = (recon_loss + beta * kld_loss).mean()

            # 1b. 新增：自洽性损失 (Consistency Loss)
            # 让分类器来判断VAE生成的图像
            logits_from_generated = classifier(recon)
            # 损失的目标是E-step给出的伪标签
            consistency_loss = F.cross_entropy(logits_from_generated, x_indices)

            # 1c. 合并生成器的总损失
            total_loss_g = vae_loss + lambda_consistency * consistency_loss
            
            total_loss_g.backward()
            optimizer_g.step()

            # --- 步骤 2: 独立训练分类器 ---
            optimizer_c.zero_grad()
            
            # 使用 .detach() 来阻断梯度流回生成器
            # 分类器的唯一任务是：学会识别生成器当前生成的东西
            logits_for_classifier = classifier(recon.detach())
            loss_c = F.cross_entropy(logits_for_classifier, x_indices)
            
            loss_c.backward()
            optimizer_c.step()

            total_g_loss += total_loss_g.item()
            total_c_loss += loss_c.item()
            num_batches += 1
            
    avg_g_loss = total_g_loss / num_batches
    avg_c_loss = total_c_loss / num_batches
    print(f"  M-Step Avg Generator Loss: {avg_g_loss:.4f}, Avg Classifier Loss: {avg_c_loss:.4f}")
    
    return new_label_priors, avg_g_loss
    
# 损失函数: Negative ELBO
def loss_function(recon_y, y, mu, logvar):
    # recon_y: (N, 1, 28, 28), y: (N, 1, 28, 28)
    BCE = F.binary_cross_entropy(recon_y, y, reduction='none').sum(dim=[1, 2, 3])
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return BCE + KLD

# --- 6. 结果评估 ---
def validation(model, epoch, test_loader, label_priors, num_classes, device, latent_dim, save_dir=SAVE_PATH):
    model.eval()
    with torch.no_grad():
        # (1) 评估聚类准确度
        # 使用 E-Step 函数来获取测试集的 gamma 分布
        test_gamma, test_true_labels = e_step(model, test_loader, label_priors, num_classes, device)
        predicted_clusters = torch.argmax(test_gamma, dim=1).cpu().numpy()
        test_true_labels = test_true_labels.numpy()

        # 建立聚类索引到真实标签的映射
        mapping = {}
        for i in range(num_classes):
            # 找到所有被预测为聚类 i 的真实标签
            true_labels_for_cluster_i = test_true_labels[predicted_clusters == i]
            if len(true_labels_for_cluster_i) > 0:
                # 将这个聚类映射到最常见的真实标签
                most_common_label = mode(true_labels_for_cluster_i, keepdims=False)[0]
                mapping[i] = most_common_label

        # 计算准确率
        correct_predictions = 0
        for i in range(len(predicted_clusters)):
            if predicted_clusters[i] in mapping:
                if mapping[predicted_clusters[i]] == test_true_labels[i]:
                    correct_predictions += 1

        accuracy = correct_predictions / len(predicted_clusters)
        print(f"[Validation] Clustering Accuracy on Test Set: {accuracy * 100:.2f}%")
        print("Cluster to Label Mapping:", mapping)

        # (2) 条件生成图像
        print("\n[Validation] Generating images conditioned on inferred labels...")
        with torch.no_grad():
            # 为每个类别生成 10 张图像
            num_gens_per_class = 10
            # 从标准正态分布中采样 z
            z_samples = torch.randn(num_gens_per_class * num_classes, latent_dim).to(device)
            
            # 创建要生成的标签
            gen_labels = torch.arange(num_classes).repeat(num_gens_per_class)
            gen_x_onehot = F.one_hot(gen_labels, num_classes).to(device).float()
            
            generated_images = model.decode(z_samples, gen_x_onehot).cpu()
            save_path = os.path.join(save_dir, f"samples/generated_samples_epoch_{epoch}.png")  
            save_image(generated_images,
               save_path,
               nrow=num_gens_per_class)
               
        print(f"[Validation] Generated images saved to {save_path}")

        return accuracy, mapping

def loss_plot(loss_history, save_path=os.path.join(SAVE_PATH, "loss_plot.png")):
    """
    绘制训练损失的图表。
    """
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(loss_history)), loss_history, marker='o', linestyle='-')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()