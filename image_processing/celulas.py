import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

# if value > 0 and value < 175: celulas muertas
# if value > 175 and value < 189: celulas vivas

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
    for i in range(x-2,x+3):
        for j in range(y-2,y+3):
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

def celulas(img,a,b):

    matrix = [ [ img.item(i,j) for j in range(img.shape[1])] for i in range(img.shape[0]) ]
    #modificando la imagen
    print(len(matrix[0]),len(matrix))
    print(img.shape[0],img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if threshold(img.item(i,j),a,b):
                matrix[i][j] = -1
                img.itemset(i,j,255)

    cv2.imshow('detectados.png', img)
    suavizar(img,matrix)


#original
img = cv2.imread('celulas2.png', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cv2.imshow('normal.png', img)

celulas(img,175,189)
new_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cv2.imshow('modified.png', img)

# plot 
plt.plot(hist, color='red' ) # original
plt.plot(new_hist, color='blue',alpha = 0.4 ) # threshold

plt.show()

cv2.destroyAllWindows()