from PIL import Image
from matplotlib import image
from numpy import asarray
from pathlib import Path
import ntpath
import os, os.path
import numpy as np

pascal_contour_color = [224, 224, 192]


def from_ann_to_instance_mask(ann, mask_outpath):
    mask = np.zeros((ann.img_size[0], ann.img_size[1], 3), dtype=np.uint8)
    for label in ann.labels:
        if label.obj_class.name == "neutral":
            label.geometry.draw(mask, pascal_contour_color)
            continue

        label.geometry.draw_contour(mask, pascal_contour_color, pascal_contour_thickness)
        label.geometry.draw(mask, label.obj_class.color)

    im = Image.fromarray(mask)
    im.convert("P", palette=Image.ADAPTIVE)
    im.save(mask_outpath)