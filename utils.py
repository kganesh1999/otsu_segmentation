import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray

def visualizePixels(numpy_arr):
  plt.imshow(numpy_arr,cmap='gray')
  plt.show()

def applyThreshold(image, threshold):
    height, width = image.shape
    flat = image.flatten()
    for i in range(len(flat)):
        if flat[i] * 255 >= threshold:
            flat[i] = 255
        else:
            flat[i] = 0
    output = flat.reshape(height, width)
    return output