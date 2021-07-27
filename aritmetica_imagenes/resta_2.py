import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def threshold(value,a,b):
    if value >= a and value <= b:
        return 1
    else:
        return 0


def function(rango_inicial,rango_final,value):
    if value < rango_inicial[0]:
        return rango_final[0]
    if value > rango_inicial[1]:
        return rango_final[1]
    
    ri_len = rango_inicial[1]-rango_inicial[0]
    rf_len = rango_final[1]-rango_final[0]
    return ((value-rango_inicial[0])*rf_len/ri_len) + rango_final[0]


# contrast streching
def contrast(img, low_percent = 0, high_percent = 1):

    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])

    #calculando el nuevo rango
    float_hist = [float(i) for i in histogram]
    indexs = []
    for i in range(len(float_hist)):
        indexs += [i] * int(float_hist[i])
    
    min_index = indexs[int((len(indexs)-1)*low_percent)]
    max_index = indexs[int((len(indexs)-1)*high_percent)]

    print(min_index,max_index)
    #modificando la imagen
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                img.itemset(i,j,function([min_index,max_index],[0,255],img[i,j]))
    
    return img


image1 = 'image7.png'
image2 = 'image8.png'

img2 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_val = (int(img[i,j]) - int(img2[i,j]))+255/2
            img[i,j] = new_val

cv2.imshow('diference.png', img)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='green',alpha=0.5 ) # original

img = contrast(img,0.05,0.95)
cv2.imshow('contrast.png', img)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='red',alpha=0.5 ) # original

for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if threshold(img[i,j],0,5) or threshold(img[i,j],251,255):
                img[i,j] = 0
            else:
                img[i,j] = 255
cv2.imshow('threshold.png', img)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='blue',alpha=0.5 ) # original

plt.show()

cv2.destroyAllWindows()

