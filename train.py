import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm
from ImagesDataset import et
from cnn import CNN
import os
import shutil
import random
from datetime import datetime
def data_split():
    source_folder = 'images'
    # 目标文件夹路径
    target_folder_1 = 'targetfolder_1_'
    target_folder_2 = 'targetfolder_2_'
    now = datetime.now()
    formatted_time = now.strftime("%H%M%S")
    target_folder_1 = target_folder_1 + formatted_time
    target_folder_2 = target_folder_2 + formatted_time
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(target_folder_1, exist_ok=True)
    os.makedirs(target_folder_2, exist_ok=True)

    images = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]


    print(len(images))
    # 随机打乱图片列表
    random.shuffle(images)

    # 根据比例4:1分割图片列表
    split_index = int(len(images) * 4 / 5)
    images_1 = images[:split_index]
    images_2 = images[split_index:]

    # 分别复制图片到两个目标文件夹
    for img in images_1:
        shutil.copy(img, target_folder_1)

    for img in images_2:
        shutil.copy(img, target_folder_2)

    return target_folder_1, target_folder_2
def main():
    device=torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    t1,t2=data_split()
    train_dataset = et(img_dir=t1, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = et(img_dir=t2, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_class=11).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # 训练的轮数

    print(torch.cuda.is_available())
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(tqdm(train_loader)):
            image ,label= image.to('cuda'),label.to('cuda')

            outputs = model(image)
            print(f"{outputs.size()}*****{label.size()}")
            loss = criterion(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochs}轮')
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

if __name__ == '__main__':
    main()