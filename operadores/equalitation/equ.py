import sys
import cv2
import numpy as np

from math import floor, log1p
from matplotlib import pyplot as plt


def main():
    if len(sys.argv) < 2:
        return
    crop = False
    if len(sys.argv) > 5:
        crop = True

    fig, axs = plt.subplots(3) if crop else plt.subplots(2)
    fig.suptitle('Histograms')

    img = cv2.imread(sys.argv[1], 0)
    height, width = img.shape

    cropA, cropB, cropC, cropD = 0, 0, height, width
    if crop:
        cropA, cropB, cropC, cropD = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    cropImg = img[cropA:cropC, cropB:cropD]
    height, width = cropImg.shape

    h, _ = np.histogram(cropImg.flatten(), bins = 256, range = (0, 256))
    h = np.cumsum(h)

    LUT = []
    factor = 255 / (width * height)
    for i in range(256):
        LUT.append(floor(factor * h[i]))

    LUT = np.array(LUT, dtype = np.uint8)
    outImg = LUT[img]

    axs[0].hist(img.flatten(), bins = 256, range = (0, 256))
    if crop:
        axs[1].hist(cropImg.flatten(), bins = 256, range = (0, 256))
        axs[2].hist(outImg.flatten(), bins = 256, range = (0, 256))
    else:
        axs[1].hist(outImg.flatten(), bins = 256, range = (0, 256))

    cv2.imshow('Input Image', img)
    cv2.imshow('Output Image', outImg)
    cv2.imwrite(f'{sys.argv[1][:-4]}-out{sys.argv[1][-4:]}', outImg)
    cv2.imwrite(f'{sys.argv[1][:-4]}-crop{sys.argv[1][-4:]}', cropImg)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
