import sys

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import random

def threshold(img,threshold):
    imgres = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if imgres[i,j] < threshold:
                imgres[i,j] = 0
            else:
                imgres[i,j] = 255
    return  imgres

def ANDfunc(p,q):
    return p | q

def ORfunc(p,q):
    return p & q

def XORfunc(p,q):
    return (p ^ q)

def imagemerge(img1,img2,func):
    imgres = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgres[i,j] = func(img1[i,j],img2[i,j])
    return  imgres


t = 120
archivo = 'img1'
archivo2 = 'img2'
img = cv2.imread(archivo+'.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(archivo2+'.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('imgs.jpg',np.concatenate([img,img2],axis=1))


avg =0

##plots
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].hist(img.flatten(),bins=256,range=(0,256))
# plt.show()

##
imgfin= []

# img 1
t = 175
imgres = threshold(img,t)
# cv2.imshow(archivo+' t = '+str(t),imgres)
imgfin.append(imgres.copy())
cv2.imwrite('img1res.jpg',imgres)

# img2
t = 176
imgres = threshold(img2,t)
# cv2.imshow(archivo2+' t = '+str(t),imgres)
imgfin.append(imgres.copy())
cv2.imwrite('img2res.jpg',imgres)

##Threshold

cv2.imwrite('imgsres.jpg',np.concatenate([imgfin[0],imgfin[1]],axis=1))

## AND
imgres = imagemerge(imgfin[0],imgfin[1],ANDfunc)
cv2.imshow('AND operation',imgres)
cv2.imwrite('AND'+'.jpg',imgres)
## OR
imgres = imagemerge(imgfin[0],imgfin[1],ORfunc)
cv2.imshow('OR operation',imgres)
cv2.imwrite('OR'+'.jpg',imgres)
## XOR
imgres = imagemerge(imgfin[0],imgfin[1],XORfunc)
cv2.imshow('XOR operation',imgres)
cv2.imwrite('XOR'+'.jpg',imgres)



finalimg = []
cv2.waitKey(0)
cv2.destroyAllWindows()