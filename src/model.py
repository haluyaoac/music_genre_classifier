import torch.nn as nn

# 简单的 CNN：输入为单通道 mel 频谱，输出类别 logits。
class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # n_classes: 分类类别数。

        # 卷积 + BN + ReLU + 池化堆叠，最后全局平均池化。
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        # 线性分类头：将 64 维特征映射到类别数。
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [batch, 1, n_mels, time]
        x = self.net(x).flatten(1)
        # 输出 logits: [batch, n_classes]
        return self.fc(x)
