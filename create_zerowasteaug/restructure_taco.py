import os
from pathlib import Path
import shutil
from shutil import copyfile
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/guest/mabdelfa/zerowaste/DRS/utils/transforms')
from PIL import Image
import imageio
import numpy
from numpy import asarray
import numpy as np


#/research/axns2/mabdelfa/before_after_voc_format/SegmentationClass
#VOC2012
#before_after_voc_format
#dataset_conversion
#/research/axns2/mabdelfa/sem_conv

directory_of_interest = "/research/axns2/mabdelfa/TACO/data/batch_"
count = 0
total = 15
total_files = 0
all_old_names = [[]]
for i in range(1, total+1):
    my_dir = directory_of_interest + str(i)
    directory = os.fsencode(my_dir)
    path, dirs, files = next(os.walk(directory))
    file_count = len(files)
    total_files += file_count
    names = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".JPG"): 
            count +=1
            names.append(filename)
            #my_file = Path(my_dir + "/" + filename)
            #save_path = "/research/axns2/mabdelfa/TACO/data/coco_format/data/" + filename
            #shutil.copy2(my_file, save_path)
    all_old_names.append(names)
print('total_files: ' , total_files)




f = open("map_names.txt", "w")
counter = 0
batch_n = -1
for batch in all_old_names:
    batch_n += 1
    batch.sort()
    for old_name in batch:
        new_name = str(counter) + '.jpg'
        if not os.path.isfile('batch_' + str(batch_n) + '/' + old_name):
            print('Big problem, this file does not exits.... ' + 'batch_' + str(batch_n) + '/' + old_name)
        f.write('batch_' + str(batch_n) + '/' + old_name + ' ' + new_name + '\n')
        counter += 1
f.close()