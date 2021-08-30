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


parts = ['train', 'test', 'val']


for part in parts:
    directory = os.fsencode("/research/axns2/mabdelfa/zerowaste/zerowaste-f/" + part+ "/data")

    #bad_images_list = [image_id.strip() for image_id in open('bad_images.txt').readlines()]

    f = open(part + ".txt", "w")




    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #print(filename)

        if filename.endswith(".PNG"): 
            name_png = os.path.splitext(filename)[0]
            f.write(name_png + "\n")
            print("Writing... " +  name_png + " to " + part)

    f.close()