from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


part = 'test'

save_dir = '/research/axns2/mabdelfa/TACO/data/zerowaste_taco_aug/' + part + '/sem_seg/'
annFile = '/research/axns2/mabdelfa/TACO/data/zerowaste_taco_aug/' + part + '/labels.json'
#annFile = '/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/labels.json'
coco=COCO(annFile)
cat_ids = coco.getCatIds()
#catIds = coco.getCatIds(catNms=['metal'])
catIds = coco.getCatIds()


imgIds = coco.getImgIds( )



for id in imgIds:
    imgIds = coco.getImgIds(imgIds = id)

    
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)

    map_name = img['file_name']
    save_here = save_dir + map_name

    print('Generating... ' + map_name)
    
    
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img['height'],img['width']))
    for ann in anns:
        #print(coco.annToMask(ann))
        if not ann['segmentation']:
            continue
        anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])


    anns_img = anns_img.astype(int)

    
    #plt.imsave(save_here, anns_img)
    
    cv2.imwrite(save_here, anns_img)

    """ my_img = Image.open(save_here)

    image_sequence = my_img.getdata()
    image_array = np.array(image_sequence) """



