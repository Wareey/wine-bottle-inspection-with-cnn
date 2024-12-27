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
from PIL import Image
import numpy as np
def is_size(image_path):
    with Image.open(image_path) as img:
        return img.size == (658,492)
def data_split():
    source_folder = 'images'
    # 目标文件夹路径
    target_folder_1 = 'targetfolder_1_'
    target_folder_2 = 'targetfolder_2_'
    now = datetime.now()
    formatted_time = now.strftime("%H%M%S")
    target_folder_1A = target_folder_1 + formatted_time +"A"
    target_folder_1B = target_folder_1 + formatted_time +"B"
    target_folder_2A = target_folder_2 + formatted_time +"A"
    target_folder_2B = target_folder_2 + formatted_time +"B"
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(target_folder_1A, exist_ok=True)
    os.makedirs(target_folder_1B, exist_ok=True)
    os.makedirs(target_folder_2A, exist_ok=True)
    os.makedirs(target_folder_2B, exist_ok=True)

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
        if is_size(img):
            shutil.copy(img, target_folder_1A)
        else:
            shutil.copy(img, target_folder_1B)

    for img in images_2:
        if is_size(img):
            shutil.copy(img, target_folder_2A)
        else:
            shutil.copy(img, target_folder_2B)


    return target_folder_1A, target_folder_1B,target_folder_2A,target_folder_2B
def main():
    device=torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    t1A,t1B,t2A,t2B=data_split()
    #t1A为瓶盖训练集
    #t1B为标签训练集

    #t2A为瓶盖测试集
    #t2B为标签测试集

    train_dataset6=et(img_dir=t1B,transform=transform,ca_id=6)
    train_loader6 = DataLoader(train_dataset6, batch_size=32, shuffle=True)
    test_dataset6=et(img_dir=t2B,transform=transform,ca_id=6)
    test_loader6 = DataLoader(test_dataset6, batch_size=32, shuffle=True)
    model6=CNN(num_class=2).to(device)

    train_dataset7=et(img_dir=t1B,transform=transform,ca_id=7)
    train_loader7 = DataLoader(train_dataset7, batch_size=32, shuffle=True)
    test_dataset7=et(img_dir=t2B,transform=transform,ca_id=7)
    test_loader7 = DataLoader(test_dataset7, batch_size=32, shuffle=True)
    model7=CNN(num_class=2).to(device)

    train_dataset8=et(img_dir=t1B,transform=transform,ca_id=8)
    train_loader8 = DataLoader(train_dataset8, batch_size=32, shuffle=True)
    test_dataset8=et(img_dir=t2B,transform=transform,ca_id=8)
    test_loader8 = DataLoader(test_dataset8, batch_size=32, shuffle=True)
    model8=CNN(num_class=2).to(device)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model6.parameters(), lr=0.001)

    num_epochs = 1 # 训练的轮数

    print(torch.cuda.is_available())
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(tqdm(train_loader6)):
            # image ,label= image.to('cuda'),label.to('cuda')

            outputs = model6(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader6)}], Loss: {loss.item():.4f}')
    
    model6.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader6:
            images, labels = data
            outputs = model6(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochs}轮')
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

if __name__ == '__main__':
    main()
