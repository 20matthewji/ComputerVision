#!/usr/bin/python

import numpy as np
import pylab
import mahotas as mh

dir='/Users/haimingjin/Pictures/Strt/'
n='Park_Ave'
## Direct image cropping
from PIL import Image
from PIL import ImageFilter
from random import randint

im.mode => "RGB" "L" "CMYK"
im.size => w * h
cropim.show()
def preproc(f):
    im = Image.open(f)
    box=0,0,800,800
    size=128,128
    cropim=im.crop(box)
    cropim.thumbnail(size, Image.ANTIALIAS)
    cropim.save(dir+f+".thumb.png", "PNG")
    grayim=cropim.convert("L")
    grayim.save(dir+n+".gray.png", "PNG")
    edgeim=cropim.filter(ImageFilter.FIND_EDGES)
    edgeim.save(dir+f+".edge.png", "PNG")

# Process Image with Numpy 
f = '/Users/haimingjin/Pictures/Strt/Park_Ave.gray.png'
img = mh.imread(f)

# Get rid of bright sky noise
T = mh.thresholding.otsu(img)

pylab.imshow(img>T)

h0,w0 = img.shape

mh.imsave('fixedimage.png', img)
gray2d = np.asarray(im)	## 3D array
# constructing Mahotas image array from 2D gray image: 2D is show as negative in pylab.imshow(gray2d)
def gray2img3d(gray2d):
    h,w=gray2d.shape
    gray3d=np.array([[[0]*3]*w]*h, dtype=uint8); ## usigned is important as otherwise it may show as negative
    for i in range(h):
        for j in range(w):
            gray3d[i][j]=np.array([gray2d[i][j]]*3, dtype=uint8)
    return gray3d

np.array([col for col in img if len(col)<=900])
np.array([row for row in img if len(row)<=
def crop(h,w,img):
  for i in range(h):
    for j in range(w):
      cropimg[i][j]=img[i][j]
  return cropimg
    
def rm_artifact(x0,y0,x1,y1,img):
  for i in range(y0:y1):
    for j in range(x0:x1):
      if all(img[i][j]>170) :
         img[i][j]=np.array([randint(75,85), randint(80,90), randint(85,95)])
  return img
    
#plylab.gray()
#plylab.show()

## Find boundary of the image:
pix = np.asarray(im)	## 3D array
pix = pix[:,:,0:3] # Drop alpha channel
pix = 255 - pix # Invert the image
H = pix.sum(axis=2).sum(axis=1)	#sum colors, then the y-axis
