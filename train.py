import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm
from ImagesDataset import et
from cnn import CNN


device=torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])
train_dataset = et(img_dir=r'Z:\CNN\images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 假设您已经有了数据加载器 train_loader
# for inputs, labels in train_loader:
#     inputs = inputs.cuda()
#     labels = labels.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_class=11).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2  # 训练的轮数

print(torch.cuda.is_available())
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        image ,label= image.to('cuda'),label.to('cuda')
        
        outputs = model(image)
        print(f"{outputs.size()}*****{label.size()}")
        loss = criterion(outputs,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

