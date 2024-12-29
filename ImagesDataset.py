from image import image
from annotation import annotation
import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class et(Dataset):
    def __init__(self, img_dir,ca_id,transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.ca_id=ca_id
    
    def __len__(self):
        return len(self.img_filenames)
    
    def load_data_from_json(self,file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            images = [image(p['file_name'], p['id'],p['height'],p['width']) for p in data['images']]
            annotations = [annotation(p['image_id'], p['bbox'],p['iscrowd'],p['area'],p['category_id'],p['id']) for p in data['annotations']]
            
        return images,annotations
    def __getitem__(self, idx):

        images_list,annotations_list=self.load_data_from_json(r'annotations.json')

        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        
        for i in images_list:
            if i.file_name ==self.img_filenames[idx]:
                id=i.id
                break
        label=0
        for i in annotations_list:
            if i.image_id==id and i.category_id==self.ca_id:
                label=1
            elif label==1:
                break
                # print("来自函数__getitem__")
                # print(f"********************************{label}******************")

                

        if self.transform:
            image = self.transform(image)

        return image, label