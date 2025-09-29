import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from torch.utils.data import random_split
from VAE_config import *
from VAE_model import ConditionalVAE_CrossAttention, Classifier
from VAE_EM import e_step, m_step_contrastive, validation, loss_plot, m_step_with_classifier
from tqdm import tqdm

# 创建输出目录
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(os.path.join(SAVE_PATH, 'loss')):
    os.makedirs(os.path.join(SAVE_PATH, 'loss'))
if not os.path.exists(os.path.join(SAVE_PATH, 'model')):
    os.makedirs(os.path.join(SAVE_PATH, 'model'))
if not os.path.exists(os.path.join(SAVE_PATH, 'samples')):
    os.makedirs(os.path.join(SAVE_PATH, 'samples'))


# --- 2. 数据加载 ---
full_train_loader = DataLoader(
    datasets.MNIST('./', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)
full_train_dataset_tensors = next(iter(DataLoader(full_train_loader.dataset, batch_size=len(full_train_loader.dataset))))

# 获取完整的训练数据集
train_images, train_labels = full_train_dataset_tensors
train_images = train_images.to(DEVICE)

# 将测试集划分为验证集和测试集
data_loader = DataLoader(
    datasets.MNIST('./', train=False,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)

# 计算验证集和测试集的大小
total_size = len(data_loader.dataset)
val_size = int(total_size * 0.5)
test_size = total_size - val_size
# 随机划分
val_dataset, test_dataset = random_split(data_loader.dataset, [val_size, test_size])
# 验证集和测试集的DataLoader
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. 训练主循环 (最终版) ---

# 1. 实例化所有模型和优化器
# 保持使用您最强的交叉注意力VAE
model_g = ConditionalVAE_CrossAttention(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
model_c = Classifier(num_classes=NUM_CLASSES).to(DEVICE) # 新增分类器

optimizer_g = optim.Adam(model_g.parameters(), lr=LEARNING_RATE)
optimizer_c = optim.Adam(model_c.parameters(), lr=LEARNING_RATE) # 新增分类器优化器

# 2. EM算法和超参数
label_priors = torch.ones(NUM_CLASSES, device=DEVICE) / NUM_CLASSES
loss_history = []
beta_anneal_epochs = 40  # 保持较长的退火



for em_epoch in range(NUM_EM_EPOCHS):
    print(f"\n--- EM Epoch {em_epoch + 1}/{NUM_EM_EPOCHS} ---")
    current_beta = min(1.0, em_epoch / beta_anneal_epochs)
    print(f"Current Beta: {current_beta:.4f}")

    # E-Step (不变, 保持使用 e_step_full_elbo)
    gamma, _ = e_step(model_g, full_train_loader, label_priors, NUM_CLASSES, DEVICE, current_beta)
    
    # M-Step: 调用新的、带有分类器约束的训练函数
    new_label_priors, epoch_loss = m_step_with_classifier(
        generator=model_g,
        classifier=model_c,
        optimizer_g=optimizer_g,
        optimizer_c=optimizer_c,
        dataloader=TensorDataset(full_train_dataset_tensors[0], full_train_dataset_tensors[1]),
        gamma=gamma,
        beta=current_beta,
        lambda_consistency=lambda_consistency
    )
                                                 
    loss_history.append(epoch_loss)
    # 验证
    if (em_epoch + 1) % VALIDATION_INTERVAL == 0:
        validation(model_g, em_epoch + 1, val_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
        # 绘制loss图表
        loss_plot(loss_history, os.path.join(SAVE_PATH, f"loss/loss_plot_epoch_{em_epoch + 1}.png"))
        # 打印先验概率的变化
        print(f"Old priors: {[f'{p:.3f}' for p in label_priors.cpu().numpy()]}")
        print(f"New priors: {[f'{p:.3f}' for p in new_label_priors.cpu().numpy()]}")

    label_priors = new_label_priors.to(DEVICE)



    # ... (后续的验证和保存逻辑不变，但注意将 model_g 传入 validation 函数) ...
    # 例如： validation(model_g, ...)

# # =================================================================
# #  == 新增部分：第一阶段 - 无条件预训练函数 ==
# # =================================================================
# def pretrain_unconditional_vae(model, optimizer, train_loader, epochs=10):
#     """
#     训练一个标准的、无条件的VAE，为EM算法提供一个好的权重起点。
#     """
#     print("\n--- Starting Phase 1: Unconditional Pre-training ---")
#     model.train()
    
#     # 临时修改模型，使其在预训练时忽略标签x
#     # 我们通过创建一个固定的dummy_x来实现这一点
#     dummy_x = torch.zeros(BATCH_SIZE, NUM_CLASSES, device=DEVICE)

#     for epoch in range(epochs):
#         total_loss = 0
#         for y_batch, _ in tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{epochs}"):
#             y_batch = y_batch.to(DEVICE)
            
#             # 如果最后一批次大小不匹配，调整dummy_x
#             if y_batch.size(0) != BATCH_SIZE:
#                 dummy_x = torch.zeros(y_batch.size(0), NUM_CLASSES, device=DEVICE)

#             optimizer.zero_grad()
            
#             # 使用一个固定的、不起作用的dummy_x进行前向传播
#             recon, mu, logvar = model(y_batch, dummy_x)
            
#             # 标准的VAE损失 (beta=1.0)
#             recon_loss = F.binary_cross_entropy(recon, y_batch, reduction='sum')
#             kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#             loss = (recon_loss + kld_loss) / y_batch.size(0) # 按样本数归一化
            
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
            
#         print(f"Pre-train Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
    
#     print("--- Unconditional Pre-training Finished ---\n")
    

# 1. 模型和优化器 (不变)
# 建议使用我们之前定义的交叉注意力模型
# model = ConditionalVAE_CrossAttention(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # =================================================================
# #  == 新增部分：执行第一阶段 ==
# # =================================================================
# # 加载普通的训练数据
# pretrain_loader = DataLoader(
#     datasets.MNIST('./', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=BATCH_SIZE, shuffle=True)
# # 执行预训练，例如10个epochs
# pretrain_unconditional_vae(model, optimizer, pretrain_loader, epochs=10)


# print("\n--- Starting Phase 2: Conditional EM Fine-tuning ---")
# # =================================================================


# --- 5. 训练主循环 ---
# # load model
# sup_load_path = 'results_em_cvae_supervised'
# model_path = os.path.join(sup_load_path, 'model/cvae_mnist.pth')
# model = ConditionalVAE_CNN(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
# try:
#     model.load_state_dict(torch.load(model_path, weights_only=False, map_location=DEVICE))
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()
    
# model = ConditionalVAE_CrossAttention(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # 初始化类别先验为均匀分布
# label_priors = torch.ones(NUM_CLASSES, device=DEVICE) / NUM_CLASSES

# # 记录loss
# loss_history = []


# for em_epoch in range(NUM_EM_EPOCHS):
#     print(f"\n--- EM Epoch {em_epoch + 1}/{NUM_EM_EPOCHS} ---")

#     # 计算当前epoch的beta值
#     # current_beta = 1.0
#     current_beta = min(1.0, em_epoch / beta_anneal_epochs)
#     print(f"Current Beta: {current_beta:.4f}")
#     # E-Step
#     gamma, _ = e_step(model, full_train_loader, label_priors, NUM_CLASSES, DEVICE)
    
#     # M-Step
#     new_label_priors, epoch_loss = m_step_final(model, optimizer, 
#                                                   TensorDataset(full_train_dataset_tensors[0],
#                                                                 full_train_dataset_tensors[1]), 
#                                                   gamma, 
#                                                   current_beta)
                                                  
#     loss_history.append(epoch_loss)

#     # 验证
#     if (em_epoch + 1) % VALIDATION_INTERVAL == 0:
#         validation(model, em_epoch + 1, val_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
#         # 绘制loss图表
#         loss_plot(loss_history, os.path.join(SAVE_PATH, f"loss/loss_plot_epoch_{em_epoch + 1}.png"))
#         # 打印先验概率的变化
#         print(f"Old priors: {[f'{p:.3f}' for p in label_priors.cpu().numpy()]}")
#         print(f"New priors: {[f'{p:.3f}' for p in new_label_priors.cpu().numpy()]}")

#     label_priors = new_label_priors.to(DEVICE)

print("\n--- Training Finished ---")
# # 保存模型
# torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model/cvae_mnist.pth'))
# print("Model saved to model/cvae_mnist.pth")

# 保存loss历史记录
np.save(os.path.join(SAVE_PATH, 'model/cvae_mnist_loss_history.npy'), loss_history)
print("Loss history saved to model/cvae_mnist_loss_history.npy")

# 测试集验证
test_accuracy, _ = validation(model_g, NUM_EM_EPOCHS, test_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
print(f"[Test] Clustering Accuracy on Test Set: {test_accuracy * 100:.2f}%")


