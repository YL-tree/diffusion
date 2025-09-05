import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================================================
# 假设您的代码已经组织在以下文件中，请根据您的实际情况修改导入
# ===============================================================
from diffusion import forward_add_noise_pair, posterior_probs # 从您的 diffusion.py 导入
from unet import UNet
from mnist_data import MNIST

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K = 10 # 类别数

# ===============================================================
# 验证主程序
# ===============================================================
def validate_classification_at_t(model, val_loader, t_values):
    """
    在给定的时间步 t_values 列表上验证模型的分类准确性。

    :param model: 训练好的 U-Net 模型
    :param val_loader: 验证数据加载器
    :param t_values: 一个包含要测试的时间步 t 的列表
    :return: 一个字典，key 是时间步 t，value 是对应的准确率
    """
    model.eval()  # 将模型设置为评估模式
    accuracies = {}

    with torch.no_grad(): # 在评估时不需要计算梯度
        for t_val in tqdm(t_values, desc="Evaluating at different timesteps"):
            correct_predictions = 0
            total_samples = 0
            
            # 为当前 t 值固定时间步
            t = torch.full((val_loader.batch_size,), t_val, device=DEVICE).long()

            for imgs, labels in val_loader:
                # 准备数据
                x0 = imgs.to(DEVICE) * 2 - 1
                true_labels = labels.to(DEVICE)
                
                # 如果最后一个批次大小不匹配，需要调整 t 的大小
                if x0.size(0) != t.size(0):
                    t = torch.full((x0.size(0),), t_val, device=DEVICE).long()

                # 1. 前向加噪过程
                z_t, z_tm1, eps_eff = forward_add_noise_pair(x0, t)

                # 2. 使用模型计算后验概率
                # 注意：这里我们不需要 tau 或者 log_prior，因为我们只关心 argmax
                probs = posterior_probs(model, z_t, z_tm1, t, K=10, eps_eff=eps_eff)

                # 3. 得到预测的类别
                # torch.argmax 在 dim=1 上操作，找到每个样本概率最高的类别索引
                predicted_labels = torch.argmax(probs, dim=1)

                # 4. 统计正确预测的数量
                correct_predictions += (predicted_labels == true_labels).sum().item()
                total_samples += x0.size(0)

            # 计算当前 t 值的总准确率
            accuracy = correct_predictions / total_samples
            accuracies[t_val] = accuracy
            print(f"Timestep t = {t_val}, Accuracy = {accuracy:.4f}")

    return accuracies

def plot_accuracy_vs_t(accuracies):
    """
    绘制准确率随时间步 t 变化的图表。
    """
    t_values = sorted(accuracies.keys())
    acc_values = [accuracies[t] for t in t_values]

    plt.figure(figsize=(12, 7))
    plt.plot(t_values, acc_values, marker='o', linestyle='-')
    plt.title('Model Classification Accuracy vs. Diffusion Timestep (t)')
    plt.xlabel('Timestep (t)')
    plt.ylabel('Prediction Accuracy')
    plt.grid(True)
    # y轴范围设为0到1，更直观
    plt.ylim(0, 1.05)
    # 在x=0附近显示1/K的随机猜测准确率线
    plt.axhline(y=1/K, color='r', linestyle='--', label=f'Random Guess (1/{K})')
    plt.legend()
    plt.savefig('accuracy_vs_t.png')
    plt.show()


if __name__ == '__main__':
    # ==================
    # 1. 参数设置
    # ==================
    BATCH_SIZE = 256  
    MODEL_PATH = 'model/unet.pth' 

    # ==================
    # 2. 加载模型
    # ==================
    model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4),
                 time_emb_dim=128, num_classes=K).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # ==================
    # 3. 准备验证数据
    # ==================
    val_dataset = MNIST(is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ==================
    # 4. 定义要测试的时间步
    # ==================
    # 我们选择一系列 t 值，从接近 0 到接近 T
    # np.linspace 比手动写列表更方便
    t_to_validate = np.linspace(1, T - 1, num=20, dtype=int)
    # 也可以手动指定: t_to_validate = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

    # ==================
    # 5. 执行验证并绘图
    # ==================
    accuracy_results = validate_classification_at_t(model, val_loader, t_to_validate)
    plot_accuracy_vs_t(accuracy_results)