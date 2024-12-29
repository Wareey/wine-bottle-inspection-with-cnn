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

def is_size(image_path):
    with Image.open(image_path) as img:
        return img.size == (658,492)
def data_split():#分割数据集
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
    learning_rate = 0.001  # 学习率
    fanzhuanP=0.66  # 随机水平翻转
    num_epochs = 15  # 训练的轮数

    learning_rateA = 0.00088  # 学习率
    fanzhuanPA=0.66  # 随机水平翻转
    num_epochsA = 9  # 训练的轮数

    device=torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=fanzhuanP),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    transformA= transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.RandomHorizontalFlip(p=fanzhuanPA),  # 随机水平翻转
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    t1A,t1B,t2A,t2B=data_split()
    #t1A为瓶盖训练集
    #t1B为标签训练集

    #t2A为瓶盖测试集
    #t2B为标签测试集
    train_dataset1=et(img_dir=t1A,transform=transformA,ca_id=1)
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    test_dataset1=et(img_dir=t2A,transform=transformA,ca_id=1)
    test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=True)
    model1 = CNN(num_class=2).to(device)

    train_dataset2=et(img_dir=t1A,transform=transformA,ca_id=2)
    train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)
    test_dataset2=et(img_dir=t2A,transform=transformA,ca_id=2)
    test_loader2 = DataLoader(test_dataset2, batch_size=32, shuffle=True)
    model2 = CNN(num_class=2).to(device)

    train_dataset3=et(img_dir=t1A,transform=transformA,ca_id=3)
    train_loader3 = DataLoader(train_dataset3, batch_size=32, shuffle=True)
    test_dataset3=et(img_dir=t2A,transform=transformA,ca_id=3)
    test_loader3 = DataLoader(test_dataset3, batch_size=32, shuffle=True)
    model3 = CNN(num_class=2).to(device)

    train_dataset4=et(img_dir=t1A,transform=transformA,ca_id=4)
    train_loader4 = DataLoader(train_dataset4, batch_size=32, shuffle=True)
    test_dataset4=et(img_dir=t2A,transform=transformA,ca_id=4)
    test_loader4 = DataLoader(test_dataset4, batch_size=32, shuffle=True)
    model4 = CNN(num_class=2).to(device)

    train_dataset5=et(img_dir=t1A,transform=transformA,ca_id=5)
    train_loader5 = DataLoader(train_dataset5, batch_size=32, shuffle=True)
    test_dataset5=et(img_dir=t2A,transform=transformA,ca_id=5)
    test_loader5 = DataLoader(test_dataset5, batch_size=32, shuffle=True)
    model5 = CNN(num_class=2).to(device)

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

    train_dataset9=et(img_dir=t1A,transform=transformA,ca_id=9)
    train_loader9 = DataLoader(train_dataset9, batch_size=32, shuffle=True)
    test_dataset9=et(img_dir=t2A,transform=transformA,ca_id=9)
    test_loader9 = DataLoader(test_dataset9, batch_size=32, shuffle=True)
    model9=CNN(num_class=2).to(device)

    train_dataset10=et(img_dir=t1A,transform=transformA,ca_id=10)
    train_loader10 = DataLoader(train_dataset10, batch_size=32, shuffle=True)
    test_dataset10=et(img_dir=t2A,transform=transformA,ca_id=10)
    test_loader10 = DataLoader(test_dataset10, batch_size=32, shuffle=True)
    model10=CNN(num_class=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
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
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader1)}], Loss: {loss.item():.4f}')
    
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
    print(f'训练了{num_epochsA}轮')
    print('类型1 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A1= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
        for i, (image, label) in enumerate(tqdm(train_loader2)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model2(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader2)}], Loss: {loss.item():.4f}')
    
    model2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader2:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochsA}轮')
    print('类型2 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A2= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
        for i, (image, label) in enumerate(tqdm(train_loader3)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model3(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader3)}], Loss: {loss.item():.4f}')
    
    model3.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader3:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model3(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochsA}轮')
    print('类型3 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A3= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model4.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
        for i, (image, label) in enumerate(tqdm(train_loader4)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model4(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader4)}], Loss: {loss.item():.4f}')
    
    model4.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader4:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model4(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochsA}轮')
    print('类型4 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A4= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model5.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
        for i, (image, label) in enumerate(tqdm(train_loader5)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model5(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader5)}], Loss: {loss.item():.4f}')
    
    model5.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader5:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model5(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochsA}轮')
    print('类型5 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A5= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model6.parameters(), lr=learning_rate)
    print(torch.cuda.is_available())
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(tqdm(train_loader6)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

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
            images, labels = images.to(device), labels.to(device)
            outputs = model6(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochs}轮')
    print('类型6 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A6= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model7.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(tqdm(train_loader7)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model7(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader7)}], Loss: {loss.item():.4f}')
    
    model7.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader7:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model7(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochs}轮')
    print('类型 7 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A7 = 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model8.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(tqdm(train_loader8)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model8(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader8)}], Loss: {loss.item():.4f}')
    
    model8.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader8:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model8(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochs}轮')
    print('类型 8 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A8= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model9.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
        for i, (image, label) in enumerate(tqdm(train_loader9)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model9(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader9)}], Loss: {loss.item():.4f}')
    
    model9.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader9:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model9(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochsA}轮')
    print('类型9 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A9= 100 * correct / total

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model10.parameters(), lr=learning_rateA)
    for epoch in range(num_epochsA):
        for i, (image, label) in enumerate(tqdm(train_loader10)):
            # image ,label= image.to('cuda'),label.to('cuda')
            image ,label= image.to(device),label.to(device)

            outputs = model10(image)
            # print(f"{output6.size()}*****{label.size()}")
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochsA}], Step [{i+1}/{len(train_loader10)}], Loss: {loss.item():.4f}')
    
    model10.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader10:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model10(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'训练了{num_epochsA}轮')
    print('类型10 Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    A10= 100 * correct / total


    print(f'标贴模型学习率为{learning_rate} 训练了{num_epochs}轮  随机翻转率{fanzhuanP}')
    print(f'瑕疵6 测试集准确率{A6}')
    print(f'瑕疵7 测试集准确率{A7}')
    print(f'瑕疵8 测试集准确率{A8}')
    print(f'瓶盖模型学习率为{learning_rateA} 训练了{num_epochsA}轮  随机翻转率{fanzhuanPA}')
    print(f'瑕疵1 测试集准确率{A1}')
    print(f'瑕疵2 测试集准确率{A2}')
    print(f'瑕疵3 测试集准确率{A3}')
    print(f'瑕疵4 测试集准确率{A4}')
    print(f'瑕疵5 测试集准确率{A5}')
    print(f'瑕疵9 测试集准确率{A9}')
    print(f'瑕疵10 测试集准确率{A10}')
    
    
    logging.basicConfig(
    filename='train.log',  # 日志文件名
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )
    logging.info(f'标贴模型学习率为{learning_rate} 训练了{num_epochs}轮  随机翻转率{fanzhuanP}')
    logging.info(f'瑕疵6 测试集准确率{A6}')
    logging.info(f'瑕疵7 测试集准确率{A7}')
    logging.info(f'瑕疵8 测试集准确率{A8}')
    logging.info(f'瓶盖模型学习率为{learning_rateA} 训练了{num_epochsA}轮  随机翻转率{fanzhuanPA}')
    logging.info(f'瑕疵1 测试集准确率{A1}')
    logging.info(f'瑕疵2 测试集准确率{A2}')
    logging.info(f'瑕疵3 测试集准确率{A3}')
    logging.info(f'瑕疵4 测试集准确率{A4}')
    logging.info(f'瑕疵5 测试集准确率{A5}')
    logging.info(f'瑕疵9 测试集准确率{A9}')
    logging.info(f'瑕疵10 测试集准确率{A10}')

if __name__ == '__main__':
    main()
