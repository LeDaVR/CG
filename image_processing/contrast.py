import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

# escala un valor de un rango a otro
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
    


#original
img = cv2.imread('ruido.png', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cv2.imshow('normal.png', img)

#llamada al contrast
contrast(img,0.05,0.95)
hist3 = cv2.calcHist([img], [0], None, [256], [0, 256])
cv2.imshow('modify.png', img)


# plot 
plt.plot(hist, color='gray' ) # original
plt.plot(hist3, color='blue',alpha=0.4 ) # modificado
plt.show()

cv2.destroyAllWindows()
