import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from otsuThresholding import OTSU
from utils import visualizePixels

if __name__ == '__main__':
    img = mpimg.imread('coins.jpg')
    # Convert to a grayscale image.
    gray_img = rgb2gray(img)
    image_size = gray_img.shape[0] * gray_img.shape[1]
    # Show the grayscale image.
    visualizePixels(gray_img)
    otsu1 = OTSU(gray_img)
    otsu1.viewHistogram()
    optimal_threshold = otsu1.getThreshold()
    print("OTSU Threshold for Image 1 : ", optimal_threshold)
    otsu1.getOutput()