import os
from pathlib import Path
import shutil
from shutil import copyfile
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/guest/mabdelfa/zerowaste/DRS/utils/transforms')
from PIL import Image
import torch
import imageio
import numpy
from numpy import asarray
import numpy as np


#/research/axns2/mabdelfa/before_after_voc_format/SegmentationClass
#VOC2012
#before_after_voc_format
#dataset_conversion
#/research/axns2/mabdelfa/sem_conv

directory_of_interest = "/research/axns2/mabdelfa/before_after_voc_format/JPEGImages"


directory = os.fsencode(directory_of_interest)
count = 0

for file in os.listdir(directory):
    count +=1
    filename = os.fsdecode(file)

    my_file = Path(directory_of_interest + "/" + filename)

    save_path = "/research/axns2/mabdelfa/before_after_voc_format/temp/" + filename

    gt_map = Image.open(my_file)

    gt_map = gt_map.resize((500,350))

    data = asarray(gt_map)

    print ('Image: ', data.shape, gt_map.mode, count)

    gt_map.save(save_path)