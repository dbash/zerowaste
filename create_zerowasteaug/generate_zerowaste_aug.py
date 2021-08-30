import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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

random.seed(10)
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
    filename = obj_name

    ironman = Image.open(filename, 'r')
    
    if resize_flag:
        ironman = ironman.resize((h,w))

    if flip:
        ironman = ImageOps.flip(ironman)

    if mirror:
        ironman = ImageOps.mirror(ironman)
    #ironman = ironman.rotate(random_rot)

    enhancer = ImageEnhance.Brightness(ironman)
    ironman = enhancer.enhance(0.9)

    #ironman = ironman.filter(ImageFilter.GaussianBlur(2.5))

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
    mask = obj_name
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


def get_and_resize_new_seg(resized_mask_name, img_id, ann_id, category_id):
    ground_truth_binary_mask = cv2.imread(resized_mask_name, 0)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": category_id,
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


def determine_resize_param(obj_name, raw_obj_path):

    filename = raw_obj_path

    obj_on_bg = obj_name

    im_bg = Image.open(obj_on_bg, 'r')
    im = Image.open(filename, 'r')
    h, w = im.size

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


metal_id = 3
rigid_plastic_id = 1
part = 'test'
old_dataset = '/research/axns2/mabdelfa/zerowaste/zerowaste-f/'
#new_dataset = '/research/axns2/mabdelfa/ZeroWasteAug/' 
new_dataset = '/research/axns2/mabdelfa/zero_test/' 
part_path = old_dataset+ part + '/data/'
new_path = new_dataset + part + '/data/'
old_labels_path = old_dataset + part + '/labels.json'
new_labels_path = new_dataset + part + '/labels.json'
aug_record_path = new_dataset + part + '_aug_record.json'

raw_metal_objs_dir = '/research/axns2/mabdelfa/TACO/data/metal_1_cropped_objs/'
metal_objs_dir = '/research/axns2/mabdelfa/TACO/data/metal_2_bg_size/'

raw_plastic_objs_dir = '/research/axns2/mabdelfa/TACO/data/plastic_1_cropped_objs/'
plastic_objs_dir = '/research/axns2/mabdelfa/TACO/data/plastic_2_bg_size/'

plastic_masks_dir = '/research/axns2/mabdelfa/TACO/data/plastic_3_binary_masks/'
metal_masks_dir = '/research/axns2/mabdelfa/TACO/data/metal_3_binary_masks/'

f = open(old_labels_path,)
old_labels = json.load(f)

last_ann = old_labels['annotations'][-1]

ann_id = last_ann['id'] + 1

train_list = [image_id.strip() for image_id in open(part + '.txt').readlines()]

#images in train = 1245, we need to augment 800 images

#images in val = 312, we need to augment 200 images

#images in tes = 317, we need to augment 200 images

if part == 'train':
    random_img_idxs = random.sample(range(0, 1244), 800)

    random_plastic_img_idxs = random.sample(range(0, 1244), 1200)

elif part == 'val':
    random_img_idxs = random.sample(range(0, 311), 200)

    random_plastic_img_idxs = random.sample(range(0, 311), 250)

elif part == 'test':
    random_img_idxs = random.sample(range(0, 316), 200)

    random_plastic_img_idxs = random.sample(range(0, 316), 200)

cc = 0

sel_imgs = []
used_objs = []

for idx in random_img_idxs:
    sel_imgs.append(train_list[idx])

sel_plastic_imgs = []

for idx in random_plastic_img_idxs:
    sel_plastic_imgs.append(train_list[idx])


augmentations_record = []
counter = 0
def augment(img, img_path, save_path, augment_with, ann_id):

    if augment_with == 'metal':
        raw_obj_dir  = raw_metal_objs_dir
        obj_dir = metal_objs_dir
        mask_dir = metal_masks_dir
        category_id = 3
        num_of_objs = random.randint(1, 3)
        random_obj_idxs = random.sample(range(0, 550), num_of_objs)

    elif augment_with == 'plastic':
        raw_obj_dir  = raw_plastic_objs_dir
        obj_dir = plastic_objs_dir
        mask_dir = plastic_masks_dir
        category_id = 1
        num_of_objs = random.randint(1, 1)
        random_obj_idxs = random.sample(range(0, 513), num_of_objs)
    

    i = 0
    for idx in random_obj_idxs:      
        obj_name = obj_dir + str(idx) + '.png'
        raw_obj_path = raw_obj_dir + str(idx) + '.png'
        mask_path = mask_dir + str(idx) + '.png'

        print('C&P ' + obj_name + ' on ' + img + ' ' + str(counter) + '/'+ str(len(train_list)))

        x_shift, y_shift, h, w, resize_flag, flip, mirror = determine_resize_param(obj_name, raw_obj_path)
        x_shift = y_shift = 0
        if i > 0:
            img_path = save_path
        x_shift = y_shift = 0
        #
        #print(x_shift, y_shift)
        paste_and_resize_obj_on_img(obj_name, img_path, save_path, x_shift, y_shift, h, w, resize_flag, flip, mirror)
        get_and_resize_old_mask(mask_path, img_path, "resized_mask.png", x_shift, y_shift, h, w, resize_flag, flip, mirror)
        
        img_info = next(item for item in old_labels['images'] if item["file_name"] == img + '.PNG')
        new_annotation = get_and_resize_new_seg("resized_mask.png", img_info['id'], ann_id, category_id)
        old_labels['annotations'].append(new_annotation)
        ann_id += 1
        #validate_new_seg(save_path, "check_seg.png") 

        x = {
            "image": img,
            "obj": obj_name,
            "category_id": category_id,
            "resize_flag": resize_flag,
            "flip": flip,
            "mirror": mirror,
            "w": w,
            "h": h
        }

        augmentations_record.append(x)

        i+=1

    return ann_id


for img in train_list:

    counter += 1

    img_path = part_path + img + '.PNG'
    save_path = new_path + img + '.PNG'
    if img in sel_imgs and img in sel_plastic_imgs:
        augment_with = 'metal'
        ann_id = augment(img, img_path, save_path, augment_with, ann_id)
        augment_with = 'plastic'
        img_path = save_path
        ann_id = augment(img, img_path, save_path, augment_with, ann_id)

    elif img in sel_imgs:
        augment_with = 'metal'
        ann_id = augment(img, img_path, save_path, augment_with, ann_id)

    elif img in sel_plastic_imgs:
        augment_with = 'plastic'
        ann_id = augment(img, img_path, save_path, augment_with, ann_id)

    else:

        #print('Copying... ', img)
        shutil.copy2(img_path, save_path)


with open(new_labels_path, "w") as outfile: 
    json.dump(old_labels, outfile, indent = 4)

with open(aug_record_path, "w") as outfile: 
    json.dump(augmentations_record, outfile, indent = 4)

print('Done!')
