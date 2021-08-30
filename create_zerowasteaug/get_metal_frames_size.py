import json
import os
import shutil
import ntpath
from PIL import Image

f = open('remember_objs.json',)
data = json.load(f)

img_dir = './coco_format/data/'

for obj_info in data:
    image_name = obj_info['org_img']
    img_path = img_dir + image_name

    img = Image.open(img_path, 'r')

    print(img.size)