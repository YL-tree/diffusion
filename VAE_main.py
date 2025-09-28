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
from VAE_model import ConditionalVAE_CrossAttention
from VAE_EM import e_step, m_step_contrastive, validation, loss_plot


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
    
model = ConditionalVAE_CrossAttention(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化类别先验为均匀分布
label_priors = torch.ones(NUM_CLASSES, device=DEVICE) / NUM_CLASSES

# 记录loss
loss_history = []
# beta 从0线性增长到1，假设在前50个EM epoch完成
beta_anneal_epochs = 20 

for em_epoch in range(NUM_EM_EPOCHS):
    print(f"\n--- EM Epoch {em_epoch + 1}/{NUM_EM_EPOCHS} ---")

    # 计算当前epoch的beta值
    # current_beta = 1.0
    current_beta = min(1.0, em_epoch / beta_anneal_epochs)
    print(f"Current Beta: {current_beta:.4f}")
    # E-Step
    gamma = e_step(model, full_train_loader, label_priors, NUM_CLASSES, DEVICE)
    
    # M-Step
    new_label_priors, epoch_loss = m_step_contrastive(model, optimizer, 
                                                  TensorDataset(full_train_dataset_tensors[0],
                                                                full_train_dataset_tensors[1]), 
                                                  gamma, 
                                                  current_beta)
                                                  
    loss_history.append(epoch_loss)

    # 验证
    if (em_epoch + 1) % VALIDATION_INTERVAL == 0:
        validation(model, em_epoch + 1, val_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
        # 绘制loss图表
        loss_plot(loss_history, os.path.join(SAVE_PATH, f"loss/loss_plot_epoch_{em_epoch + 1}.png"))
        # 打印先验概率的变化
        print(f"Old priors: {[f'{p:.3f}' for p in label_priors.cpu().numpy()]}")
        print(f"New priors: {[f'{p:.3f}' for p in new_label_priors.cpu().numpy()]}")

    label_priors = new_label_priors.to(DEVICE)

print("\n--- Training Finished ---")
# 保存模型
torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'model/cvae_mnist.pth'))
print("Model saved to model/cvae_mnist.pth")

# 保存loss历史记录
np.save(os.path.join(SAVE_PATH, 'model/cvae_mnist_loss_history.npy'), loss_history)
print("Loss history saved to model/cvae_mnist_loss_history.npy")

# 测试集验证
test_accuracy, _ = validation(model, NUM_EM_EPOCHS, test_loader, label_priors, NUM_CLASSES, DEVICE, LATENT_DIM)
print(f"[Test] Clustering Accuracy on Test Set: {test_accuracy * 100:.2f}%")


