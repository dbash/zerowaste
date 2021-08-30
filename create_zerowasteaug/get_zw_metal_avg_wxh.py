import json
import os
import shutil
import ntpath


#
f = open('/research/axns2/mabdelfa/TACO/data/coco_format/labels.json',)
data = json.load(f)
total_w = 0
total_h = 0
n = 0
for ann in data['annotations']:
    if ann['category_id'] == 3:
        total_w += ann['bbox'][2]
        total_h += ann['bbox'][3]
        n += 1
print(total_w/n, total_h/n)

f = open('/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/labels.json',)
data = json.load(f)
total_w = 0
total_h = 0
n = 0
for ann in data['annotations']:
    if ann['category_id'] == 3:
        total_w += ann['bbox'][2]
        total_h += ann['bbox'][3]
        n += 1
print(total_w/n, total_h/n)