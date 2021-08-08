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


directory = os.fsencode("/research/axns2/mabdelfa/dataset_conversion/SegmentationClass")
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print('converting... ',filename)

    image_path = "/research/axns2/mabdelfa/dataset_conversion/SegmentationClass/" + filename

    gt_map = Image.open(image_path)

    gt_map = gt_map.convert('P')

    save_path = "/research/axns2/mabdelfa/dataset_conversion/temp/" + filename

    gt_map.save(save_path)

     