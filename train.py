# from config import *
from torch.utils.data import DataLoader
from mnist_data import MNIST
from diffusion import forward_add_noise
import torch
from torch import nn
import os
from dit import DiT
from tqdm import tqdm  # 新增导入
import matplotlib.pyplot as plt  # 新增导入
from unet import UNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

dataset = MNIST()  # 数据集

model = DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(DEVICE)
# model = UNet(img_channels=1, base_ch=64, channel_mults=(1, 2, 4), time_emb_dim=128, num_classes=10).to(DEVICE)

try:  # 加载模型
    model.load_state_dict(torch.load('model/transformer_model.pth'))
except:
    pass

optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器
loss_fn = nn.L1Loss()  # 损失函数(绝对值误差均值)


# 初始化损失图保存函数
def save_loss_plot(loss_list, plot_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    # EPOCH = 100
    EPOCH = 500
    BATCH_SIZE = 300
    T = 1000
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, persistent_workers=True)
    model_path = 'model/transformer_model.pth'
    loss_plot_path = 'results/transformer_loss.png'
    model.train()
    iter_count = 0
    # 初始化损失列表
    loss_list = []
    for epoch in tqdm(range(EPOCH), desc='Training', unit='epoch'):  # 添加进度条
        for imgs, labels in dataloader:
            x = imgs * 2 - 1  # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
            t = torch.randint(0, T, (imgs.size(0),))  # 为每张图片生成随机t时刻
            y = labels

            x, noise = forward_add_noise(x, t)  # x:加噪图 noise:噪音
            pred_noise = model(x.to(DEVICE), t.to(DEVICE), y.to(DEVICE))

            loss = loss_fn(pred_noise, noise.to(DEVICE))
            # 记录损失
            loss_list.append(loss.item())

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            if iter_count % 1000 == 0:
                print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss))
                torch.save(model.state_dict(), model_path)
                os.replace(model_path, model_path)
            iter_count += 1

    # 训练完成后，保存损失图
    save_loss_plot(loss_list, loss_plot_path)