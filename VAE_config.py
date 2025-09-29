import torch

# --- 1. 参数设置 ---
# EM 训练参数
NUM_EM_EPOCHS = 50   # EM算法迭代次数
NUM_M_STEP_EPOCHS = 2 # 每次M-step中，模型训练的轮数

# 模型参数
LATENT_DIM = 18       # 潜变量z的维度
NUM_CLASSES = 10      # 类别数量 (MNIST有10类)

# 数据和训练参数
BATCH_SIZE = 100
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_INTERVAL = 5 
SAVE_PATH = 'results_em_cvae_classify'

lambda_contrast = 0.1
lambda_diversity = 0.5
# 新增：自洽性损失的权重
lambda_consistency = 1.0 # 这是一个可以调整的关键超参数

# beta 从0线性增长到1，假设在前50个EM epoch完成
beta_anneal_epochs = 40 