import json
import os
import shutil
import ntpath
import numpy as np
import cv2
from PIL import Image

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
  
f = open('./coco_format/labels.json',)
data = json.load(f)


remember_list = []

metal = [0,1,2,8,10,11,12,28,50,52]

count = 0

for annotation in data['annotations']:
    #print('hello')
    if annotation['category_id'] == 3:
        original_seg = annotation['segmentation']
        original_bbox = annotation['bbox']
        image_id = annotation['image_id']

        dicts = data['images']
        image_name = next(item for item in dicts if item["id"] == image_id)

        original_w = image_name['width']
        original_h = image_name['height']
        file_name = image_name['file_name']
        image_name = './coco_format/data/' + file_name

        

        for seg in original_seg:
            my_segmentation = np.array(seg)

            my_segmentation = my_segmentation.flatten()


            print('Getting object from: ' + file_name + ' Seg id: ' + str(annotation['id']) + ' length: ', len(my_segmentation))

            my_segmentation = np.reshape(my_segmentation, (-1, 2))


            img = cv2.imread(image_name)
            pts = np.array(my_segmentation)

            ## (1) Crop the bounding rect
            rect = cv2.boundingRect(pts)
            x,y,w,h = rect
            croped = img[y:y+h, x:x+w].copy()

            ## (2) make mask
            pts = pts - pts.min(axis=0)

            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            ## (3) do bit-op
            dst = cv2.bitwise_and(croped, croped, mask=mask)

            ## (4) add the white background
            bg = np.ones_like(croped, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst


            """ cv2.imwrite("croped.png", croped)
            cv2.imwrite("mask.png", mask)
            cv2.imwrite("bbg.png", dst) """
            cv2.imwrite("wbg.png", dst2)

            #print(image_name)

            obj_name = str(count) + '.png'

            save_path = './cropped_objs/' + obj_name

            #print('Generating... ', obj_name)

            trans = convertImage("wbg.png" , save_path)

            x = {  
                "obj": obj_name,  
                "org_img": file_name,  
                "org_img_w": original_w,
                "org_img_h": original_h,
                "org_bbox": original_bbox,
                "org_ann": original_seg
                }  


            remember_list.append(x)
            count += 1


with open("remember_objs.json", "w") as outfile: 
    json.dump(remember_list, outfile, indent=4)