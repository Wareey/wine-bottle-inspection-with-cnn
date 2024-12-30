import os
import shutil
import random
from train import is_size
def data_split():#分割数据集
    source_folder = 'images'
    # 目标文件夹路径
    target_folder_1 = 'TTrainFolder'
    target_folder_2 = 'TestFolder'

    target_folder_1A = target_folder_1 + "A"
    target_folder_1B = target_folder_1 + "B"
    target_folder_2A = target_folder_2 + "A"
    target_folder_2B = target_folder_2 + "B"
    #  A为瓶盖，B为标贴
    #1为训练集，2为测试集

    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(target_folder_1A, exist_ok=True)
    os.makedirs(target_folder_1B, exist_ok=True)
    os.makedirs(target_folder_2A, exist_ok=True)
    os.makedirs(target_folder_2B, exist_ok=True)
    images = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]


    print(len(images))
    # 随机打乱图片列表
    random.shuffle(images)

    # 15%测试集  15%验证集 70%训练集
    split_index = int(len(images) * 17/ 20)
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
def data_split_again(source_folder):
    if source_folder == 'TTrainFolderA':
        target_folder_1 = 'TrainFolderA'
        target_folder_2 = 'YanZhengFolderA'
    elif source_folder == 'TTrainFolderB':
        target_folder_1 = 'TrainFolderB'
        target_folder_2 = 'YanZhengFolderB'
    else:
        print("输入错误")
    #  A为瓶盖，B为标贴
    #1为训练集，2为验证集
    
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(target_folder_1, exist_ok=True)
    os.makedirs(target_folder_2, exist_ok=True)

    images = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]


    print(len(images))
    # 随机打乱图片列表
    random.shuffle(images)

    # 15%测试集  15%验证集 70%训练集
    split_index = int(len(images) * 14/ 17)
    images_1 = images[:split_index]
    images_2 = images[split_index:]

    # 分别复制图片到两个目标文件夹
    for img in images_1:
            shutil.copy(img, target_folder_1)


    for img in images_2:
            shutil.copy(img, target_folder_2)





if __name__ == '__main__':
    data_split()
    data_split_again('TTrainFolderA')
    data_split_again('TTrainFolderB')
