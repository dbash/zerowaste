from PIL import Image
from matplotlib import image
from numpy import asarray
from pathlib import Path
import ntpath
import os, os.path
import numpy as np


def changeColor(image_file):

    im = Image.open(image_file).convert('RGB')

    data = np.array(im)

    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
    mask = (red == 1) & (green == 1) & (blue == 1)
    data[:,:,:3][mask] = [20, 140, 220]

    mask = (red == 2) & (green == 2) & (blue == 2)
    data[:,:,:3][mask] = [180, 10, 120]

    mask = (red == 3) & (green == 3) & (blue == 3)
    data[:,:,:3][mask] = [150, 200, 20]

    mask = (red == 4) & (green == 4) & (blue == 4)
    data[:,:,:3][mask] = [10, 200, 60]
    
    #image_file = image_file.replace(".png","")
    im = Image.fromarray(data)
    #im.save(f'folder_name/{image_file}.png')
    im.save('rgb.png')
    
    return "Done"

#my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-f/val/sem_seg/01_frame_000700.PNG")
my_file = Path("/research/axns2/mabdelfa/before_after_voc_format/SegmentationClass/04_frame_003610.png")

gt_map = Image.open(my_file)

#changeColor(my_file)

data = asarray(gt_map)

print ('original sem_seg image: ',np.unique(data), data.shape, gt_map.mode)

#gt_map = gt_map.convert('P')

#gt_map.save("P.png")

############################################################

""" gt_map = Image.open(my_file).convert('P')

n = np.array(gt_map)

n[n==1] = 200

n[n==2] = 200

n[n==3] = 200

n[n==4] = 200

n[n!=200] = 1

r = Image.fromarray(n,mode='P') 

print('P image: ',np.unique(n), n.shape,gt_map.mode)

gt_map = gt_map.getpalette()

r.putpalette(gt_map) 

r.save('result.png')  """



################################################################

""" gt_map = Image.open('rgb.png').convert('RGB')

data = asarray(gt_map)

print('rgb.png image: ' ,np.unique(data), data.shape, gt_map.mode) """

################################################################

my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/coco_mask/02_frame_001620.jpg")

gt_map = Image.open(my_file)

data = asarray(gt_map)

print ('VOC image: ',np.unique(data), data.shape, gt_map.mode)