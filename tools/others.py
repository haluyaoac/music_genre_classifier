import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCLSNet(nn.Module):
    def __init__(self, 
                 in_channels=1,          # 音频频谱图通道数：单通道（灰度）为1，多通道为3
                 freq_dim=128,           # 频谱图频率维度（定长定频固定，如梅尔频谱128维）
                 time_dim=100,           # 频谱图时间维度（定长音频对应固定时间步）
                 num_classes=10,         # 音频风格分类数（按需修改）
                 cnn_hidden_dims=[64, 128, 256], # CNN每层卷积输出通道数
                 lstm_hidden_dim=256,    # LSTM隐藏层维度
                 num_heads=4,            # 多头注意力的头数
                 fc_hidden_dim=512):     # 全连接层隐藏维度
        super(AudioCLSNet, self).__init__()
        
        # ------------------- 模块1：CNN 空间特征融合 -------------------
        self.cnn_layers = self._build_cnn_layers(in_channels, cnn_hidden_dims)
        # 计算CNN输出后的特征维度（定长定频下固定，提前计算避免动态推导）
        self.cnn_out_dim = self._calc_cnn_out_dim(in_channels, freq_dim, time_dim)
        # CNN输出 → 适配LSTM输入：转为 [批次, 时间步, 特征维度]，这里取CNN输出的时间维度为LSTM时间步
        self.lstm_time_steps = self.cnn_out_dim // lstm_hidden_dim  # 时序步长适配
        
        # ------------------- 模块2：多头+LSTM 时序特征融合 -------------------
        self.lstm = nn.LSTM(input_size=lstm_hidden_dim, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True)  # 双向LSTM增强时序捕捉
        # 多头自注意力（对LSTM输出做多头时序融合）
        self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_dim*2,  # 双向LSTM输出维度翻倍
                                                     num_heads=num_heads,
                                                     batch_first=True)
        self.attn_dropout = nn.Dropout(0.2)  # 防止过拟合
        
        # ------------------- 模块3：全连接+Softmax 分类 -------------------
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_dim*2, fc_hidden_dim),  # 输入为双向LSTM+注意力融合后的维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_hidden_dim, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    # 构建CNN层：卷积+BN+ReLU+池化，逐级提取空间特征
    def _build_cnn_layers(self, in_channels, hidden_dims):
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),  # 加速收敛，防止过拟合
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样，缩小特征图尺寸
            ])
            in_channels = dim
        return nn.Sequential(*layers)
    
    # 计算CNN输出的特征维度（定长定频下一次计算即可）
    def _calc_cnn_out_dim(self, in_channels, freq_dim, time_dim):
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, freq_dim, time_dim)  # 虚拟输入
            out = self.cnn_layers(dummy)
            return out.size(1) * out.size(2) * out.size(3)  # 展平后总维度
    
    def forward(self, x):
        # x 输入格式：[batch_size, in_channels, freq_dim, time_dim] → 对应频谱图
        batch_size = x.shape[0]
        
        # 1. CNN空间特征提取：输出 → [batch_size, cnn_out_channels, freq_out, time_out]
        cnn_feat = self.cnn_layers(x)
        # 展平：空间特征 → 时序特征序列 [batch_size, lstm_time_steps, lstm_hidden_dim]
        cnn_feat_flat = cnn_feat.view(batch_size, self.lstm_time_steps, -1)
        
        # 2. LSTM基础时序特征提取：输出 → [batch_size, lstm_time_steps, 2*lstm_hidden_dim]
        lstm_feat, _ = self.lstm(cnn_feat_flat)
        # 多头注意力时序融合：聚焦关键时序节点，输出同维度融合特征
        attn_feat, _ = self.multihead_attn(lstm_feat, lstm_feat, lstm_feat)  # q=k=v
        attn_feat = self.attn_dropout(attn_feat)
        
        # 3. 时序特征聚合：取最后一个时间步的特征（全局时序总结）
        seq_last_feat = attn_feat[:, -1, :]
        
        # 4. 全连接+Softmax分类：输出分类概率
        fc_out = self.fc_layers(seq_last_feat)
        prob = self.softmax(fc_out)
        return prob, fc_out  # 返回概率+未激活的logits（训练用logits，推理用prob）
    
# 初始化网络（适配 单通道、128×100频谱图、10分类音频风格）
net = AudioCLSNet(in_channels=1, freq_dim=128, time_dim=100, num_classes=10)
# 虚拟输入（批次16，对应16个定长定频音频的频谱图）
dummy_input = torch.randn(16, 1, 128, 100)
# 前向传播
prob, logits = net(dummy_input)
print("分类概率维度：", prob.shape)  # torch.Size([16, 10])
print("分类logits维度：", logits.shape)  # torch.Size([16, 10])