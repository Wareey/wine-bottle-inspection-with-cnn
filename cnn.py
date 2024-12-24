import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,    # 输入通道数
                out_channels=16,    # 输出通道数
                kernel_size=5,    # 卷积核大小
                stride=1,    # 步长
                padding=1    
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=16,    # 输入通道数
                out_channels=32,    # 输出通道数
                kernel_size=5,    # 卷积核大小
                stride=1,    # 步长
                padding=1    
            ),
            nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        
