import json
from image import image
from annotation import annotation
#统计数据集数据
with open('annotations.json', 'r') as file:
    data = json.load(file)
    images = [image(p['file_name'], p['id'],p['height'],p['width']) for p in data['images']]
    annotations = [annotation(p['image_id'], p['bbox'],p['iscrowd'],p['area'],p['category_id'],p['id']) for p in data['annotations']]
     
def count_elements(lst):
    element_count = {}
    for element in lst:
        if element.image_id in element_count:
            element_count[element.image_id] += 1
        else:
            element_count[element.image_id] = 1
    return element_count


# 示例使用
example_list=annotations
a=count_elements(example_list)
print()
# for i in images:
#     print(f'{i.id}高度{i.height}宽度{i.width}')
for i in a:
    for j in images:
        if i==j.id and i>4000:
            print(f'{i}有{a[i]}个目标高度{j.height}宽度{j.width}')