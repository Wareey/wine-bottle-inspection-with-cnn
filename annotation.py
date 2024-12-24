class annotation(object):
    def __init__(self, image_id:int, bbox:list[float],iscrowd:int,area:float,category_id:int,id:int):
        self.image_id=image_id
        self.bbox = bbox
        self.iscrowd=iscrowd
        self.area=area
        self.category_id=category_id
        self.id=id