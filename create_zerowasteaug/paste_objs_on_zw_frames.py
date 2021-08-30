import numpy as np
import cv2
from PIL import Image
import json
import math
from shapely import geometry

def shrink_or_swell_shapely_polygon(my_polygon, factor=0.10, swell=False):
    ''' returns the shapely polygon which is smaller or bigger by passed factor.
        If swell = True , then it returns bigger polygon, else smaller '''
    

    #my_polygon = mask2poly['geometry'][120]
    my_polygon = geometry.Polygon(my_polygon)
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*factor 

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance) #expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance) #shrink

    #visualize for debugging
    #x, y = my_polygon.exterior.xy
    #plt.plot(x,y)
    #x, y = my_polygon_shrunken.exterior.xy
    #plt.plot(x,y)
    ## to net let the image be distorted along the axis
    #plt.axis('equal')
    #plt.show()    
    
    return my_polygon_resized

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

""" img = trans
img_w, img_h = img.size
background = Image.open('after.png')
bg_w, bg_h = background.size
offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
background.paste(img, offset)
background.save('out.png') """

obj_name = "0.png"
filename = './cropped_objs/' + obj_name
ironman = Image.open(filename, 'r')#.resize((200,150))

obj_w, obj_h = ironman.size


""" basewidth = 421
wpercent = (basewidth/float(ironman.size[0]))
hsize = int((float(ironman.size[1])*float(wpercent)))
ironman = ironman.resize((basewidth,hsize), Image.ANTIALIAS) """


""" new_w = math.floor(obj_w * 0.5)
new_h = math.floor(obj_h * 0.5)
ironman = ironman.resize((new_w, new_h)) """

filename1 = 'after.png'
bg = Image.open(filename1, 'r')

h, w = bg.size
text_img = Image.new('RGBA', (h,w), (0, 0, 0, 0))
text_img.paste(bg, (0,0))
text_img.paste(ironman, (0,0), mask=ironman)

text_img.paste(bg, ((text_img.width - bg.width) // 2, (text_img.height - bg.height) // 2))
#text_img.paste(ironman, ((text_img.width - ironman.width) // 2, (text_img.height - ironman.height) // 2), mask=ironman)

f = open('remember_objs.json',)
data = json.load(f)

obj_info = next(item for item in data if item["obj"] == obj_name)
old_x = obj_info['org_bbox'][0]
old_y = obj_info['org_bbox'][1]
org_img_w = obj_info['org_img_w']
org_img_h = obj_info['org_img_h']

new_x = (old_x * w)//org_img_w
new_y = (old_y * h)//org_img_h

text_img.paste(ironman, (new_x, new_y), mask=ironman)

text_img.save("out.png", format="png")

""" 
"bbox" : [x,y,width,height]
"org_bbox": [
            589,
            548,
            341,
            405
        ], "org_img_w": 2049,
        "org_img_h": 1537,
"""

"""
x,y (old) = 589, 548
x,y (new) = 261, 516
"""

shift_x = new_x - old_x
shift_y = new_y - old_y

print(new_x, new_y)

seg =  obj_info['org_ann'][0]

def resize_seg(lst):
    lista = []
    for i, e in enumerate(lst):
        if i % 2 != 0:
            lista.append(e + shift_y)
        else:
            lista.append(e +shift_x)
    return lista

my_segmentation = resize_seg(seg)

print(len(my_segmentation))

o = [(my_segmentation[i],my_segmentation[i+1]) for i in range(0,len(my_segmentation),2)]

print(len(o))

my_segmentation = shrink_or_swell_shapely_polygon(o, 0.2)

x, y = my_segmentation.exterior.xy

result = [None]*(len(x)+len(y))
result[::2] = x
result[1::2] = y

result = list(map(int, result))

my_segmentation = np.array(result)

print(my_segmentation)


my_segmentation = my_segmentation.flatten()

#print(my_segmentation)

my_segmentation = np.reshape(my_segmentation, (-1, 2))


img = cv2.imread("out.png")
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

trans = convertImage("wbg.png" , "cropped_out_resized.png")