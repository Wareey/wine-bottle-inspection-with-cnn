import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,num_class):
        super(CNN,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
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

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()   # 继承__init__功能
#         ## 第一层卷积
#         self.conv1 = nn.Sequential(
#             # 输入[1,28,28]
#             nn.Conv2d(
#                 in_channels=3,    # 输入图片的高度
#                 out_channels=16,  # 输出图片的高度
#                 kernel_size=5,    # 5x5的卷积核，相当于过滤器
#                 stride=1,         # 卷积核在图上滑动，每隔一个扫一次
#                 padding=2,        # 给图外边补上0
#             ),
#             # 经过卷积层 输出[16,28,28] 传入池化层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)   # 经过池化 输出[16,14,14] 传入下一个卷积
#         )
#         ## 第二层卷积
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,    # 同上
#                 out_channels=32,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2
#             ),
#             # 经过卷积 输出[32, 14, 14] 传入池化层
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,7,7] 传入输出层
#         )
#         ## 输出层
#         self.output = nn.Linear(in_features=32*7*7, out_features=10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)           # [batch, 32,7,7]
#         x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
#         output = self.output(x)     # 输出[50,10]
#         return output
