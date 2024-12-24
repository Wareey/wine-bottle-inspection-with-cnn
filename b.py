import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义一个卷积层，输入通道为1（灰度图），输出通道为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 定义一个池化层，大小为2x2
        self.pool = nn.MaxPool2d(2, 2)
        # 定义一个卷积层，输入通道为16，输出通道为32，卷积核大小为3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 定义一个全连接层，输入尺寸需要根据前面层的输出尺寸确定，这里假设输入是32*7*7（取决于输入图像大小和池化层），输出是10（假设有10个类别）
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # 应用卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 应用第二个卷积层和池化层
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图，为全连接层准备
        x = x.view(-1, 32 * 7 * 7)
        # 应用全连接层
        x = self.fc1(x)
        return x

# 实例化网络
net = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络
for epoch in range(2):  # 遍历数据集多次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data

        # 梯度置零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 保存模型
PATH = './cnn.pth'
torch.save(net.state_dict(), PATH)

# 加载模型
net = SimpleCNN()
net.load_state_dict(torch.load(PATH))

# 在测试数据上测试网络
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
