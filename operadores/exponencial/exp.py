import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


def exponential(value,c,b):
    re = c * (b**value-1)
    if re < 0:
        return 0
    if re > 255:
        return 255
    return re

def raiseto(value,c,r):
    re =c*(value**r)
    if re < 0:
        return 0
    if re > 255:
        return 255
    return re

def image_procesor(image_name,c,b,c_2,r):

    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original.png', img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='black' ) # original

    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j] = exponential(img[i,j],c,b)
                
    cv2.imshow('exponencial.png', img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='blue',alpha=0.5 ) # original

    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i,j] = raiseto(img[i,j],c_2,r)
                
    cv2.imshow('raise.png', img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color='red',alpha=0.5 ) # original



image_procesor('image2.png',20,1.01,0.03,1.6)

plt.waitforbuttonpress()
plt.clf()
image_procesor('image3.png',20,1.01,0.05,1.5)
plt.waitforbuttonpress()
plt.clf()
image_procesor('image4.png',26,1.01,0.17,1.3)


plt.show()







cv2.destroyAllWindows()