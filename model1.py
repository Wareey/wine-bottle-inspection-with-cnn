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
import logging
from train import is_size

def train(ca_id,learning_rate, fanzhuanP, num_epochs):
    # learning_rate = 0.001  # 学习率
    # fanzhuanP=0.66  # 随机水平翻转
    # num_epochs = 15  # 训练的轮数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=fanzhuanP),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    t1A,t1B,t2A,t2B='TrainFolderA','TrainFolderB','YanZhengFolderA','YanZhengFolderB'
    #t1A为瓶盖训练集
    #t1B为标签训练集

    #t2A为瓶盖测试集
    #t2B为标签测试集

    if ca_id==6 or ca_id==7 or ca_id==8:
        t1=t1B
        t2=t2B
        v='标贴'
    else:
        t1=t1A
        t2=t2A
        v='瓶盖'

    train_dataset1=et(img_dir=t1,transform=transform,ca_id=ca_id)
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    test_dataset1=et(img_dir=t2,transform=transform,ca_id=ca_id)
    test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=True)
    model1 = CNN(num_class=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(tqdm(train_loader1)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model1(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader1)}], Loss: {loss.item():.4f}')
    
    model1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader1:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochs}轮')
    print(f'类型{ca_id} Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A1= 100 * correct / total

    logging.basicConfig(
    filename=f'train{ca_id}.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'  
    )
    logging.info(f'{v}模型学习率为{learning_rate} 训练了{num_epochs}轮  随机翻转率{fanzhuanP}时')
    logging.info(f'瑕疵{ca_id} 测试集准确率{A1}')

    torch.save({
    'epoch': num_epochs,
    'model_state_dict': model1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, f'{ca_id}_{num_epochs}_{learning_rate}_{fanzhuanP}.pth')

    return f'{ca_id}_{num_epochs}_{learning_rate}_{fanzhuanP}.pth'

def TrainAgain(modelpth,learning_rate, ca_id, fanzhuanP):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(modelpth)
    model= CNN(num_class=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=fanzhuanP),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    t1A,t1B,t2A,t2B='TrainFolderA','TrainFolderB','YanZhengFolderA','YanZhengFolderB'

    if ca_id==6 or ca_id==7 or ca_id==8:
        t1=t1B
        t2=t2B
        v='标贴'
    else:
        t1=t1A
        t2=t2A
        v='瓶盖'

    train_dataset1=et(img_dir=t1,transform=transform,ca_id=ca_id)
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    test_dataset1=et(img_dir=t2,transform=transform,ca_id=ca_id)
    test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=True)

    # 继续训练第11轮
    for epoch in range(start_epoch , start_epoch +1):
        for i, (image, label) in enumerate(tqdm(train_loader1)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{start_epoch + 1}], Step [{i+1}/{len(train_loader1)}], Loss: {loss.item():.4f}')
    
    model.eval()
    correct = 0
    total = 0
++++++++++++++++++++++++++++++++++++++++++++
    print(f'训练了{start_epoch + 1}轮')
    print(f'类型{ca_id} Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A1= 100 * correct / total

    logging.basicConfig(
    filename=f'train{ca_id}.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'  
    )
    logging.info(f'{v}模型学习率为{learning_rate} 训练了{start_epoch + 1}轮  随机翻转率{fanzhuanP}时')
    logging.info(f'瑕疵{ca_id} 测试集准确率{A1}')

    torch.save({
    'epoch': start_epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, f'{ca_id}_{start_epoch + 1}_{learning_rate}_{fanzhuanP}.pth')

    return f'{ca_id}_{start_epoch + 1}_{learning_rate}_{fanzhuanP}.pth'

if __name__ == '__main__':
    # for i in range(7, 15):
    #     train(ca_id=1, learning_rate=0.00088, num_epochs=i, fanzhuanP=0.5)

 #   a=TrainAgain(modelpth=train(ca_id=2, learning_rate=0.00018,  fanzhuanP=0.66,num_epochs=12),ca_id=2, learning_rate=0.00018, fanzhuanP=0.66)


    a=TrainAgain(modelpth=train(ca_id=3, learning_rate=0.003,  fanzhuanP=0.66,num_epochs=3),ca_id=3, learning_rate=0.003, fanzhuanP=0.66)
    b=TrainAgain(modelpth=a,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    c=TrainAgain(modelpth=b,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    d=TrainAgain(modelpth=c,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    e=TrainAgain(modelpth=d,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    f=TrainAgain(modelpth=e,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    g=TrainAgain(modelpth=f,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    h=TrainAgain(modelpth=g,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    i=TrainAgain(modelpth=h,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    j=TrainAgain(modelpth=i,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    k=TrainAgain(modelpth=j,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)
    l=TrainAgain(modelpth=k,ca_id=3, learning_rate=0.003,  fanzhuanP=0.66)

    