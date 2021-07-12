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

def getValue(matrix,x,y):
    if x <0 or x >=  len(matrix):
        return -1
    if y <0 or y >= len(matrix[0]):
        return -1
    return matrix[x][y]

def near_average(matrix,x,y):
    average = 0
    count = 0
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            value = getValue(matrix,i,j)
            if value < 256 and value >= 0:
                average += value
                count +=1

    return average/count

def suavizar(img,matrix):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if matrix[i][j] == -1:
                matrix[i][j] = near_average(matrix,i,j)
                img.itemset(i,j,matrix[i][j])

def campos(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            [b,g,r] = img[i,j]
            trigo = True
            if not threshold(b,g*0.75,g*0.88):
                trigo = False
            if not threshold(r,170,255):
                trigo = False
            if not threshold(g,r*0.70,r):
                trigo = False

            if trigo:
                img[i,j] = [0,0,0]
            else:
                img[i,j] = [255,255,255]

    cv2.imshow('detectados.png', img)

    matrix = [ [ img[i,j][0] for j in range(img.shape[1])] for i in range(img.shape[0]) ]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            value = near_average(matrix,i,j)
            img[i,j] = [value,value,value]

    cv2.imshow('suavizado.png', img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not threshold(img[i][j][0],100,180):
                img[i,j] = [255,255,255]
            else:
                img[i,j] = [0,0,0]
    
    cv2.imshow('bordes.png', img)
    #suavizar(img,matrix)


#original
img = cv2.imread('trigo.png')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
cv2.imshow('normal.png', img)
print(img.shape)
campos(img)
#threshold(img)
#new_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
#cv2.imshow('modified.png', img)

# plot 
#plt.plot(hist, color='red' ) # original
#plt.plot(hist, color='blue',alpha = 0.4 ) # threshold

plt.show()

cv2.destroyAllWindows()