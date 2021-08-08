# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import xml.etree.ElementTree as ET
import os, os.path
from pathlib import Path
import shutil
from shutil import copyfile
import os, os.path
from pathlib import Path
import shutil
from shutil import copyfile
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/guest/mabdelfa/zerowaste/DRS/utils/transforms')
from PIL import Image
import imageio
import numpy




def read_xml(xml_path, look_for):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    bboxes = []
    classes = []

    count = 0
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')

        if label == look_for:
            count += 1
        
        bbox_xmin = max(min(int(bbox.find('xmin').text.split('.')[0]), image_width - 1), 0)
        bbox_ymin = max(min(int(bbox.find('ymin').text.split('.')[0]), image_height - 1), 0)
        bbox_xmax = max(min(int(bbox.find('xmax').text.split('.')[0]), image_width - 1), 0)
        bbox_ymax = max(min(int(bbox.find('ymax').text.split('.')[0]), image_height - 1), 0)

        if (bbox_xmax - bbox_xmin) == 0 or (bbox_ymax - bbox_ymin) == 0:
            continue
        
        bboxes.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
        classes.append(label)
    
    return bboxes, classes, count

xml_dir = "/research/axns2/mabdelfa/dataset_conversion/Annotations/"
image_id_list = [image_id.strip() for image_id in open('val.txt').readlines()]
look_for_cat = ["rigid_plastic", "cardboard", "metal", "soft_plastic"]


totals = []
bad_images = []
for look_for in look_for_cat:
    total = 0
    count = 0
    for image_id in image_id_list:
        #print(id)
        _, tags, count = read_xml(xml_dir + image_id + '.xml', look_for)

        
        total += count
        
        """ matching = [s for s in tags if "train" in s]
        if matching:
            count += 1 """

    totals.append(total)
    #print (look_for, total)


for i in range(len(look_for_cat)):
    print(look_for_cat[i], totals[i])


""" f = open("bad_images.txt", "w")
for image_id in image_id_list:
    #print(id)
    _, tags, count = read_xml(xml_dir + image_id + '.xml', look_for)
    if not tags:
        bad_images.append(image_id)
        f.write(image_id + "\n")

        print("Writing... ", image_id)

f.close() """