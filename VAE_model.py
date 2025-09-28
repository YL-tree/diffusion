import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE_CrossAttention(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConditionalVAE_CrossAttention, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- 编码器 (保持不变) ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # -> (N, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (N, 64, 7, 7)
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(64 * 7 * 7 + num_classes, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # --- 解码器 (核心修改) ---
        # 1. 标签嵌入层，将one-hot标签映射到特征维度
        self.label_emb = nn.Linear(num_classes, 256)
        
        # 2. 将z映射到与标签嵌入相同的维度
        self.z_proj = nn.Linear(latent_dim, 256)

        # 3. 交叉注意力层
        # 我们将标签嵌入作为Query, 将z生成的特征作为Key和Value
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # 4. 后续解码层
        self.decoder_fc2 = nn.Linear(256, 64 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # -> (N, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2), # -> (N, 1, 28, 28)
            nn.Sigmoid()
        )

    # 编码器函数 (保持不变)
    def encode(self, y, x_onehot):
        h_conv = self.encoder_conv(y)
        h_conv_flat = h_conv.view(h_conv.size(0), -1)
        inputs = torch.cat([h_conv_flat, x_onehot], dim=1)
        h_fc = F.relu(self.encoder_fc(inputs))
        return self.fc_mu(h_fc), self.fc_logvar(h_fc)
    
    # 重参数化函数 (保持不变)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码器函数 (核心修改)
    def decode(self, z, x_onehot):
        # 将 z 和 x 分别投影到注意力维度
        z_projected = self.z_proj(z).unsqueeze(1)    # Shape: (B, 1, 256) -> 作为 K, V
        label_emb = self.label_emb(x_onehot).unsqueeze(1) # Shape: (B, 1, 256) -> 作为 Q

        # 应用交叉注意力
        # Query: 标签信息, Key/Value: 潜变量信息
        attn_output, _ = self.attention(query=label_emb, key=z_projected, value=z_projected)
        attn_output = attn_output.squeeze(1) # Shape: (B, 256)

        # 将注意力输出送入后续层
        h_fc2 = F.relu(self.decoder_fc2(attn_output))
        h_deconv_input = h_fc2.view(h_fc2.size(0), 64, 7, 7)
        return self.decoder_deconv(h_deconv_input)

    def forward(self, y, x_onehot):
        mu, logvar = self.encode(y, x_onehot)
        z = self.reparameterize(mu, logvar)
        recon_y = self.decode(z, x_onehot)
        return recon_y, mu, logvar
