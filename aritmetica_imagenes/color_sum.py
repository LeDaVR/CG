import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

image1 = 'image3.png'
image2 = 'image4.png'

img2 = cv2.imread(image1)
cv2.imshow('img1.png', img2)
hist1 = cv2.calcHist([img2], [0], None, [256], [0, 256])

img = cv2.imread(image2)
cv2.imshow('img2.png', img)
hist2 = cv2.calcHist([img], [0], None, [256], [0, 256])

for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_val = np.array([
                int(img[i,j][k]) + int(img2[i,j][k]) for k in range(3)])/2
            img[i,j] = new_val


cv2.imshow('color_sum.png', img)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.plot(hist1, color='red',alpha=0.5 ,label='img1') # 
plt.plot(hist2, color='blue',alpha=0.5 ,label='img2') # 
plt.plot(hist, color='purple',alpha=1 ,label='Sum') # 
plt.legend()
plt.show()
cv2.destroyAllWindows()