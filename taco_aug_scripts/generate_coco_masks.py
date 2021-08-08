from pycocotools.coco import COCO
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# The path of the original coco data set
dataDir = "/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/"
# The path used to save the newly generated mask data
savepath = "/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/coco_mask"

annFile = '/research/axns2/mabdelfa/zerowaste/zerowaste-f/train/labels.json'
#annFile = './coco_format/labels.json'

'''
 Data set parameters
'''
# coco has 80 classes, here write the name of the class to be binarized
# Others not written will be regarded as the background and become black
classes_names = ["rigid_plastic", "cardboard", "metal", "soft_plastic"]

datasets_list = ['train2014']


# Generate save path, copy function (›´ω`‹)
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# Generate mask map
def mask_generator(coco, width, height, anns_list):
    mask_pic = np.zeros((height, width))
    # Generate mask-Here is a 4-channel mask image. If you want to change to three-channel, you can remove the comment below, or search for related programs when using the picture to three-channel
    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask_pic += mask_single
    # Convert to 255
    for row in range(height):
        for col in range(width):
            if (mask_pic[row][col] > 0):
                mask_pic[row][col] = 255
    mask_pic = mask_pic.astype(int)
    return mask_pic

    # Convert to three channels
    # imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
    # imgs[:, :, 0] = mask_pic[:, :]
    # imgs[:, :, 1] = mask_pic[:, :]
    # imgs[:, :, 2] = mask_pic[:, :]
    # imgs = imgs.astype(np.uint8)
    # return imgs


# Process json data and save binary mask
def get_mask_data(annFile, mask_to_save):
    # Get COCO_json data
    coco = COCO(annFile)
    # Get the id of all the image data needed-what is the id of the categories I need
    classes_ids = coco.getCatIds(catNms=classes_names)
    # Take all the image ids of the union of all categories
    # If you want to intersect, you don’t need to loop, just enter all categories as parameters, and you can get pictures included in all categories
    imgIds_list = []
    # Circulate out which pictures correspond to each category id and get the id number of the picture
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)  # Put all the image ids of this category into a list
        imgIds_list += imgidx
        print("Search id...", imgidx)
    # Remove duplicate pictures
    imgIds_list = list(set(imgIds_list))  # Combine the same image id corresponding to multiple categories

    # Get all image information at once
    image_info_list = coco.loadImgs(imgIds_list)

    # Generate a mask for each picture
    for imageinfo in image_info_list:
        # Get the segmentation information of the corresponding category
        annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None)
        anns_list = coco.loadAnns(annIds)
        # Generate binary mask map
        mask_image = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
        # save Picture
        file_name = mask_to_save + '/' + imageinfo['file_name'][:-4] + '.jpg'

        #if os.path.isfile(file_name):
        #    pass
        plt.imsave(file_name, mask_image)
        print("Saved mask picture: ", file_name)


if __name__ == '__main__':
    # Process as a single data set
    for dataset in datasets_list:
    # Used to save the last generated mask image directory
    # mask_to_save = savepath +'mask_images/' + dataset # Three-channel mask image storage path
        mkr(savepath)
        mkr(savepath + 'mask_images')
        # Generate path
        
        # Get the path of the json file to be processed
        
        # Data processing
        get_mask_data(annFile, savepath)
        print('Got all the masks of {} from {}'.format(classes_names, dataset))
