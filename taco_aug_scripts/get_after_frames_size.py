#before: /research/axns2/mabdelfa/zerowaste/zerowaste-w/before
#after: /research/axns2/mabdelfa/zerowaste/zerowaste-w/after
#Segemntation maps locations:

### /research/axns2/mabdelfa/zerowaste/zerowaste-f/train/sem_seg
### /research/axns2/mabdelfa/zerowaste/zerowaste-f/val/sem_seg
### /research/axns2/mabdelfa/zerowaste/zerowaste-f/test/sem_seg


import os
from pathlib import Path
import shutil
from shutil import copyfile
from PIL import Image

dir = "/research/axns2/mabdelfa/zerowaste/zerowaste-w/org/before"
directory = os.fsencode(dir)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".PNG"): 

        img_path = dir + "/" + filename
        img = Image.open(img_path, 'r')

        print(img.size)