import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from utils import *

def getHistogram(image):
    row, col = image.shape
    histogram = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            pixel = int(image[i, j].item() * 255)
            histogram[pixel] += 1
    return histogram

def viewHistogram(image):
    pixel_values = np.arange(0, 256)
    histogram = getHistogram(image)
    plt.bar(pixel_values, histogram)
    plt.show()

if __name__=='__main__':
    img = mpimg.imread('coins.jpg')
    # Convert to a grayscale image.
    gray_img = rgb2gray(img)
    print('Input image in grayscale')
    visualizePixels(gray_img)
    print('Histogram of input grayscale image')
    viewHistogram(gray_img)
    print('Threshold binarization for value 70')
    output1 = applyThreshold(gray_img, 70)
    visualizePixels(output1)
    print('Threshold binarization for value 150')
    output2 = applyThreshold(gray_img, 150)
    visualizePixels(output2)
    print('Threshold binarization for value 220')
    output3 = applyThreshold(gray_img, 220)
    visualizePixels(output3)
