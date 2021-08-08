import numpy as np
import cv2
from PIL import Image, ImageOps
import json
import math
from pycocotools import mask
from skimage import measure
from skimage.io import imread
import matplotlib.pyplot as plt
import random
import shutil
import json 
import ntpath
import os
# Data to be written 


#1920 x 1080


def from_iterable(iterables):
    # chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
    for it in iterables:
        for element in it:
            yield element
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

def paste_and_resize_obj_on_img(obj_name, img_name, resized_img_name, x_shift, y_shift, h, w, resize_flag = False, flip= False, mirror= False):
    filename = './objs_on_bg_size/' + obj_name

    ironman = Image.open(filename, 'r')
    
    if resize_flag:
        ironman = ironman.resize((h,w))

    if flip:
        ironman = ImageOps.flip(ironman)

    if mirror:
        ironman = ImageOps.mirror(ironman)
    #ironman = ironman.rotate(random_rot)


    filename1 = img_name
    bg = Image.open(filename1, 'r')

    h, w = bg.size
    text_img = Image.new('RGBA', (h,w), (0, 0, 0, 0))
    text_img.paste(bg, (0,0))

    text_img.paste(bg, ((text_img.width - bg.width) // 2, (text_img.height - bg.height) // 2))
    text_img.paste(ironman, (x_shift,y_shift), mask=ironman)

    text_img.save(resized_img_name)
    #, format="PNG"


def get_and_resize_old_mask(obj_name, img_name, resized_mask_name, x_shift, y_shift, h, w, resize_flag = False, flip= False, mirror= False):
    mask = './zero_one_masks_on_zw_dim/' + obj_name
    resized_mask = Image.open(mask, 'r')

    if resize_flag:
        resized_mask = resized_mask.resize((h,w))

    #resized_mask = resized_mask.rotate(random_rot)
    if flip:
        resized_mask = ImageOps.flip(resized_mask)
    if mirror:
        resized_mask = ImageOps.mirror(resized_mask)


    filename1 = img_name
    bg = Image.open(filename1, 'r')
    h, w = bg.size
    text_img = Image.new('RGBA', (h,w), (0, 0, 0, 0))
    text_img.paste(resized_mask, (x_shift,y_shift))

    text_img.save(resized_mask_name, format="png")


def get_and_resize_new_seg(resized_mask_name, img_id, ann_id):
    ground_truth_binary_mask = cv2.imread(resized_mask_name, 0)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 3,
            "bbox": ground_truth_bounding_box.tolist(),
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    
    with open('mask.json', "w") as outfile: 
        json.dump(annotation, outfile, indent = 4)

    return annotation

def validate_new_seg(resized_img_name, validated_seg_name):
    f = open('mask.json',)
    data = json.load(f)

    seg = data['segmentation']

    seg = list(from_iterable(seg))

    my_segmentation = np.array(seg)



    my_segmentation = my_segmentation.flatten()

    my_segmentation = my_segmentation. astype(int)

    my_segmentation = np.reshape(my_segmentation, (-1, 2))


    img = cv2.imread(resized_img_name)
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

    trans = convertImage("wbg.png" , validated_seg_name)


def determine_resize_param(obj_name, current_obj_idx):

    filename = './cropped_objs/' + obj_name

    obj_on_bg = './objs_on_bg_size/' + obj_name

    im_bg = Image.open(obj_on_bg, 'r')
    im = Image.open(filename, 'r')
    h, w = im.size
    

    #random_rot = random.randint(1, 180)

    """ obj_ann = next(item for item in data['annotations'] if item["id"] == current_obj_idx)
    
    #"bbox" : [x,y,width,height]
    bbox = obj_ann['bbox']

    x_pos = bbox[0]
    obj_w = bbox[2]
    y_pos = bbox[1]
    obj_h = bbox[3]


    


    obj_w_space = (x_pos + obj_w)
    obj_h_space = (y_pos + obj_h)


    x_shift_range = max(bg_w - obj_w_space - obj_w,0)

    y_shift_range = max(bg_h - obj_h_space - obj_h,0) """

    #print(bbox)

    #print(x_shift_range, y_shift_range)
    x_shift = random.randint(0, 100)

    y_shift = random.randint(0, 100)

    flip = bool(random.getrandbits(1))
    mirror = bool(random.getrandbits(1))
    resize_flag = False
    bg_h = 1080
    bg_w = 1920

    if w > 500 or h > 500:
        
        new_big_h = int((400/h)*bg_h)
        new_big_w = int((500/w)*bg_w)

        size = new_big_h, new_big_w
        resize_flag = True
        try:
            #print(size)
            im_bg.thumbnail(size, Image.ANTIALIAS)
            new_big_h, new_big_w = im_bg.size
            #print (im_bg.size)
            return x_shift, y_shift, new_big_h, new_big_w, resize_flag, flip, mirror
        except IOError:
            print ("cannot create thumbnail for '%s'" % obj_name)

    


    return x_shift, y_shift, h, w, resize_flag, flip, mirror


""" paste_and_resize_obj_on_img("0.png", 'before.png', "test.png", (1000,1000), 45)

get_and_resize_old_mask("0.png" , 'before.png',"resized_mask.png", (1000,1000), 45)

get_and_resize_new_seg("resized_mask.png")

validate_new_seg("test.png", "check_seg.png") """

part = 'test'

part_path = '/research/axns2/mabdelfa/zerowaste/zerowaste-f/' + part + '/data/'

new_path = '/research/axns2/mabdelfa/TACO/data/zerowaste_taco_aug/' + part + '/data/'

old_labels_path = '/research/axns2/mabdelfa/zerowaste/zerowaste-f/' + part + '/labels.json'

new_labels_path = '/research/axns2/mabdelfa/TACO/data/zerowaste_taco_aug/' + part + '/labels.json'

f = open(old_labels_path,)
old_labels = json.load(f)

last_ann = old_labels['annotations'][-1]

ann_id = last_ann['id'] + 1

train_list = [image_id.strip() for image_id in open(part + '.txt').readlines()]

#images in train = 1245, we need to augment 800 images

#images in val = 312, we need to augment 200 images

#images in tes = 317, we need to augment 200 images

random_img_idxs = random.sample(range(0, 316), 200)

cc = 0

sel_imgs = []
used_objs = []

for idx in random_img_idxs:
    sel_imgs.append(train_list[idx])

current_obj_idx = 500



counter = 0

for img in train_list:

    counter += 1

    img_path = part_path + img + '.PNG'
    save_path = new_path + img + '.PNG'
    if img in sel_imgs:
        num_of_objs = random.randint(1, 3)
        random_obj_idxs = random.sample(range(0, 550), num_of_objs)

        for i in range(num_of_objs):
            
            obj_name = str(current_obj_idx) + '.png'

            print('C&P ' + obj_name + ' on ' + img + ' ' + str(counter) + '/'+ str(len(train_list)))

            x_shift, y_shift, h, w, resize_flag, flip, mirror = determine_resize_param(obj_name, current_obj_idx)

            x_shift = y_shift = 0
            if i > 0:
                img_path = save_path

                x_shift = y_shift = 100

            #

            #print(x_shift, y_shift)
            paste_and_resize_obj_on_img(obj_name, img_path, save_path, x_shift, y_shift, h, w, resize_flag, flip, mirror)

            get_and_resize_old_mask(obj_name, img_path, "resized_mask.png", x_shift, y_shift, h, w, resize_flag, flip, mirror)

            img_info = next(item for item in old_labels['images'] if item["file_name"] == img + '.PNG')

            new_annotation = get_and_resize_new_seg("resized_mask.png", img_info['id'], ann_id)

            old_labels['annotations'].append(new_annotation)

            ann_id += 1
            #validate_new_seg(save_path, "check_seg.png") 

            current_obj_idx += 1

            if current_obj_idx == 553:
                current_obj_idx = 0

        #break    
    else:

        #print('Copying... ', img)
        shutil.copy2(img_path, save_path)


with open(new_labels_path, "w") as outfile: 
    json.dump(old_labels, outfile, indent = 4)

print('Done!')
