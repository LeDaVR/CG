import sys

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import random

def logarithm(p,c):
    res = c * math.log10(1 + p)
    if res > 255:
        return 255
    return res

def root(p,c):
    res = c * math.sqrt(1 + p)
    if res > 255:
        return 255
    return res



def pointoperationGenerator(img, func,c,leap):
    imgres = img.copy()
    final = []
    for l in range(2):
        res = img.copy()
        row = []
        for k in range(2):
            fac = c+ k*leap + l*2*leap
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    imgres[i, j] = func(img[i, j],fac)
            print(k,c+k*leap)

            imgres = cv2.putText(imgres, 'c='+str(fac), (20,30), cv2.FONT_ITALIC ,1 , (0), 4, cv2.LINE_AA)
            imgres = cv2.putText(imgres, 'c='+str(fac), (20,30), cv2.FONT_ITALIC ,1 , (255), 2, cv2.LINE_AA)
            row.append(imgres.copy())
        res = np.concatenate(row,axis=1)
        final.append(res)
    final = np.concatenate(final,axis=0)

    return final

def pointoperation(img, func,c):
    imgres = img.copy()
    # C es una constante en la formula
    # G'(x,y) = c*log10( 1 + G(x,y) )
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgres[i, j] = func(img[i, j],c)

    return imgres

c = 60
leap = 10
archivo = 'img'+str(3)
img = cv2.imread(archivo+'.jpg', cv2.IMREAD_GRAYSCALE)

imgres = pointoperationGenerator(img, logarithm,c,leap)
print("final")
cv2.imshow(archivo,imgres)

cv2.imwrite(archivo+'res'+'.jpg',imgres)

##Root
# c = 0
# leap = 3
# imgres = pointoperationGenerator(img, root,c,leap)
# print("final")
# cv2.imshow(archivo+'root',imgres)
#
# cv2.imwrite(archivo+'root'+'res.jpg',imgres)
cv2.waitKey(0)
cv2.destroyAllWindows()