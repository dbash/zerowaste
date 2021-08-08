import os, sys
from PIL import Image

size = 200, 250


infile = 'cropped_objs/11.png'

outfile =  "resizedddd.png"
if infile != outfile:
    try:
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        h, w = im.size
        print (h, w)
        im.save(outfile, "png")
    except IOError:
        print ("cannot create thumbnail for '%s'" % infile)