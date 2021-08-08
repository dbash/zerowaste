import json
import os
import shutil
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


f = open('annotations.json',)
data = json.load(f)
  
data['categories'] = [{"id": 1,"name": "rigid_plastic","supercategory": ""},{"id": 2,"name": "cardboard","supercategory": ""},{"id": 3,"name": "metal","supercategory": ""},{"id": 4,"name": "soft_plastic","supercategory": ""}]

to_rigid_plastic = [4,5,7,21,24,27,29,43,44,45,46,47,49,54,55,56]
to_card_board = [13,14,15,16,17,18,19,20,22]
to_metal = [0,1,2,8,10,11,12,28,50,52]
to_soft_plastic = [35,36,37,38,39,40,41,42,48,58]
to_be_removed = [6,9,23,3,25,26,30,31,32,33,34,51,53,57,59]

                
to_be_deleted = []                                            
for i in range(len(data['annotations'])):
    if data['annotations'][i]["category_id"] in to_be_removed:
        to_be_deleted.append(data['annotations'][i])

for item in to_be_deleted:
    #print('Deleting... ', item)
    data['annotations'].remove(item)

for annotation in data['annotations']:
    if annotation['category_id'] in to_rigid_plastic:
        annotation['category_id'] = 1
    elif annotation['category_id'] in to_card_board:
        annotation['category_id'] = 2
    elif annotation['category_id'] in to_metal:
        annotation['category_id'] = 3
    elif annotation['category_id'] in to_soft_plastic:
        annotation['category_id'] = 4

old_paths = []
new_names = []
for line in open('map_names.txt').readlines():
    old_path = line.split()[0]
    new_name = line.split()[1]
    old_paths.append(old_path)
    new_names.append(new_name)
    #print('Reading line.... ' + line)


for name in data['images']:
    old_json_name = name['file_name']
    idx = old_paths.index(old_json_name)
    new_json_name = new_names[idx]
    name['file_name'] = new_json_name

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

f = open('data.json',)
data = json.load(f)
for annotation in data['annotations']:
    print(annotation['category_id'])