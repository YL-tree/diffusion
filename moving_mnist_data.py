import numpy as np
import imageio
from PIL import Image
import os
from tqdm import tqdm

# 使用 MNIST digits（28x28）生成移动序列帧
from torchvision.datasets import MNIST
from torchvision import transforms

def generate_moving_mnist(n_samples=10000, image_size=64, digit_size=28, num_digits=2, seq_len=20):
    dataset = MNIST(root='./', train=True, download=True)
    data = dataset.data.numpy()  # (60000, 28, 28)

    sequences = np.zeros((n_samples, seq_len, image_size, image_size), dtype=np.uint8)

    for i in tqdm(range(n_samples), desc="Generating sequences"):
        # 随机选择 num_digits 个数字
        digit_images = []
        for _ in range(num_digits):
            idx = np.random.randint(0, len(data))
            digit = data[idx]
            digit_images.append(digit)

        # 随机速度 [-1, 1]
        velocity = np.random.randint(-3, 4, size=(num_digits, 2))
        pos = np.random.randint(0, image_size - digit_size, size=(num_digits, 2))

        for t in range(seq_len):
            canvas = np.zeros((image_size, image_size), dtype=np.uint8)

            for d in range(num_digits):
                top = int(pos[d][0])
                left = int(pos[d][1])
                canvas[top:top + digit_size, left:left + digit_size] = np.maximum(
                    canvas[top:top + digit_size, left:left + digit_size],
                    digit_images[d]
                )

                pos[d] += velocity[d]

                # 反弹边界处理
                for j in [0, 1]:
                    if pos[d][j] <= 0 or pos[d][j] >= image_size - digit_size:
                        velocity[d][j] *= -1
                        pos[d][j] = np.clip(pos[d][j], 0, image_size - digit_size)

            sequences[i, t] = canvas

    return sequences

if __name__ == '__main__':
    output_path = 'moving_mnist.npy'
    if not os.path.exists(output_path):
        data = generate_moving_mnist(n_samples=10000)
        np.save(output_path, data)
        print(f"Saved dataset to {output_path}")
    else:
        print(f"{output_path} already exists.")
