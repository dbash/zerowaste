from PIL import Image
from matplotlib import image
from numpy import asarray
from pathlib import Path
import ntpath
import os, os.path

import numpy as np

import os
from pathlib import Path
import shutil
from shutil import copyfile



directory = os.fsencode("/research/axns2/mabdelfa/zerowaste/zerowaste-w/localization_maps")

count = 0 
for file in os.listdir(directory):
    
    filename = os.fsdecode(file)

    if filename.endswith(".npy"): 
        source = "/research/axns2/mabdelfa/zerowaste/zerowaste-w/localization_maps/" + filename
        data = np.load(source)

        if data.max() > 0.0:
            print(filename)
            count += 1

print(count)



""" 
path = "/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/test/before/11_frame_000961.PNG"
#ntpath.basename("/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/val/before/11_frame_000961.PNG")

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


print(os.path.splitext(path_leaf(path))[0])

my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-w/refined_pseudo_segmentation_labels/09_frame_001720.png")

gt_map = Image.open(my_file)

data = asarray(gt_map)

print (data.max())

if path.find('seg/val') != -1:
    print (path.replace('seg/val', 'org'))
elif path.find('seg/test') != -1:
    print (path.replace('seg/test', 'org')) """
