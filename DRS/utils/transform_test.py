from PIL import Image
from matplotlib import image
from numpy import asarray
from pathlib import Path

my_file = Path("/research/axns2/mabdelfa/pascal-voc/data/VOCdevkit/VOC2012/SegmentationClassAug/2008_003373.png")

gt_map = Image.open(my_file)

data = asarray(gt_map)

print ('SegmentationClassAug image shape is: ', data.shape)

my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/before/04_frame_003620.PNG")

gt_map = Image.open(my_file)

data = asarray(gt_map)

print ('seg/before image shape is: ', data.shape)

my_file = Path("/research/axns2/mabdelfa/pascal-voc/data/VOCdevkit/VOC2012/saliency_map/2008_003373.png")

gt_map = Image.open(my_file)

data = asarray(gt_map)

print ('sal image shape is: ', data.shape)

my_file = Path("/research/axns2/mabdelfa/zerowaste/zerowaste-w/seg/after/02_044000.PNG")

gt_map = Image.open(my_file)

data = asarray(gt_map)

print ('after seg shape is: ', data.shape)