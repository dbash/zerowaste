import numpy as np
import cv2
from PIL import Image
import json
import math

def from_iterable(iterables):
    # chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
    for it in iterables:
        for element in it:
            yield element
def convertImage(old_path, new_path):
    img = Image.open(old_path)
    img = img.convert("RGBA")
  
    datas = img.getdata()
  
    newData = []
  
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
  
    img.putdata(newData)
    img.save(new_path, "png")
    #print("Successful")

    return img


f = open('mask.json',)
data = json.load(f)

seg = data['segmentation']

seg = list(from_iterable(seg))

#seg = list(map(int, seg))
my_segmentation = np.array(seg)



my_segmentation = my_segmentation.flatten()

my_segmentation = my_segmentation. astype(int)

#print(my_segmentation)

my_segmentation = np.reshape(my_segmentation, (-1, 2))


img = cv2.imread("./objs_on_bg_size/100.png")
pts = np.array(my_segmentation)

## (1) Crop the bounding rect
rect = cv2.boundingRect(pts)
x,y,w,h = rect
croped = img[y:y+h, x:x+w].copy()

## (2) make mask
pts = pts - pts.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)

## (4) add the white background
bg = np.ones_like(croped, np.uint8)*255
cv2.bitwise_not(bg,bg, mask=mask)
dst2 = bg+ dst


""" cv2.imwrite("croped.png", croped)
cv2.imwrite("mask.png", mask)
cv2.imwrite("bbg.png", dst) """
cv2.imwrite("wbg.png", dst2)

#print(image_name)


#print('Generating... ', obj_name)

trans = convertImage("wbg.png" , "check_seg.png")