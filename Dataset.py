from image import image
from annotation import annotation
import json

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        images = [image(p['file_name'], p['id'],p['height'],p['width']) for p in data['images']]
        annotations = [annotation(p['image_id'], p['bbox'],p['iscrowd'],p['area'],p['category_id'],p['id']) for p in data['annotations']]
        
    return images,annotations
images_list,annotations_list=load_data_from_json(r'Z:\CNN\annotations.json')

# for i in images_list[:5]:
#     print(i.file_name)
# for i in annotations_list[:5]:
#     print(i.image_id)
