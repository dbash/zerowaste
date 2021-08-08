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


for line in open('map_names.txt').readlines():
    old_path = line.split()[0]
    new_name = line.split()[1]
    save_path = "images/" +new_name
    
    print('making... ' + new_name)
    shutil.copy2(old_path, save_path)