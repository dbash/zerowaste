import numpy as np
import cv2
from PIL import Image
import json
import math
import os

def convertImage(old_path, new_path):
    img = Image.open(old_path)
    img = img.convert("RGBA")
  
    datas = img.getdata()
  
    newData = []
  
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
  
    img.putdata(newData)
    img.save(new_path, "png")
    #print("Successful")

    return img

def resize_seg(lst):
    lista = []
    for i, e in enumerate(lst):
        if i % 2 != 0:
            lista.append(e + shift_y)
        else:
            lista.append(e +shift_x)
    return lista

dir = "/research/axns2/mabdelfa/TACO/data/cropped_objs"
save_dir = "/research/axns2/mabdelfa/TACO/data/objs_on_bg_size_with_bbox/"
directory = os.fsencode(dir)

bg_h = 1080
bg_w = 1920


labels = {
        "info": {
            "year": 2019,
            "description": "TACO",
            "date_created": "2019-12-19T16:11:15.258399+00:00"
            }
        
        }

print(type(labels))   
images = []
anns = []
my_list = []
list_of_lists = []
count = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"): 
        save_here = save_dir + filename
        """ if os.path.isfile(save_here):
            print(filename + ' already exists, I will skip it...')
            continue """
        img_path = dir + "/" + filename
        ironman = Image.open(img_path, 'r')#.resize((200,150))

        obj_w, obj_h = ironman.size

        text_img = Image.new('RGBA', (bg_w,bg_h), (0, 0, 0, 0))

        f = open('remember_objs.json',)
        data = json.load(f)

        obj_info = next(item for item in data if item["obj"] == filename)
        old_x = int(obj_info['org_bbox'][0])
        old_y = int(obj_info['org_bbox'][1])
        org_img_w = int(obj_info['org_img_w'])
        org_img_h = int(obj_info['org_img_h'])

        new_x = (old_x * bg_w)//org_img_w
        new_y = (old_y * bg_h)//org_img_h

        text_img.paste(ironman, (new_x, new_y), mask=ironman)


        #"bbox" : [x,y,width,height]
        bbox = [new_x, new_y, obj_w, obj_h]

        print('Generating.... ' , filename)

        text_img.save(save_here, format="png")

        shift_x = new_x - old_x
        shift_y = new_y - old_y

        seg =  obj_info['org_ann'][0]

        my_segmentation = resize_seg(seg)

        my_img = {
            "id": count,
            "width": bg_w,
            "height": bg_h,
            "file_name": filename
        }

        images.append(my_img)

        my_list = list(my_segmentation)
        list_of_lists.append(my_list)

        my_ann = {
            "id": count,
            "image_id": count,
            "category_id": 1,
            "segmentation": list_of_lists,
            "bbox" : bbox
        }

        list_of_lists = []
        #my_ann["segmentation"].append(my_list) 
        anns.append(my_ann)
        count += 1

labels['images'] = images

labels['annotations'] = anns

labels['categories'] = [
        {
            "id": 1,
            "name": "metal",
            "supercategory": ""
        }
]


with open(save_dir + 'labels.json', "w") as outfile: 
    json.dump(labels, outfile, indent = 4)