from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


""" category = 'plastic'

annFile = '/research/axns2/mabdelfa/TACO/data/' + category + '_2_bg_size/labels.json'
save_dir = '/research/axns2/mabdelfa/TACO/data/' + category + '_3_binary_masks/' """

annFile = '/research/axns2/mabdelfa/zero_test/test/labels.json'
save_dir = '/research/axns2/mabdelfa/zero_test/test/sem_seg/'
coco=COCO(annFile)
cat_ids = coco.getCatIds()
#catIds = coco.getCatIds(catNms=[category])
catIds = coco.getCatIds()
#imgIds = coco.getImgIds(catIds=catIds )
imgIds = coco.getImgIds()

for id in imgIds:
    imgIds = coco.getImgIds(imgIds = id)
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'],img['width']))
    for ann in anns:
        #print(coco.annToMask(ann))
        #anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])
        if ann['segmentation']:
            anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])


    anns_img = anns_img.astype(int)

    #map_name = str(id) + '.png'

    map_name = img['file_name']
    save_here = save_dir + map_name

    print('Generating... ' + map_name)
    #plt.imsave(save_here, anns_img)
    cv2.imwrite(save_here, anns_img)


