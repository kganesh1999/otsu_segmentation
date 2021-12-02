import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from utils import *

class OTSU:
  def __init__(self, image, cmap='gray'):
      self.image = image
      self.height = image.shape[0]
      self.width = image.shape[1]
      self.image_size = self.height*self.width
      self.cmap = cmap

  def getHistogram(self):
      image = self.image
      row, col = self.height,self.width
      histogram = np.zeros(256)
      for i in range(0,row):
        for j in range(0,col):
          pixel = int(image[i,j].item()*255)
          histogram[pixel] += 1
      return histogram

  def viewHistogram(self):
      image = self.image
      pixel_values = np.arange(0,256)
      histogram = self.getHistogram()
      plt.bar(pixel_values,histogram)
      plt.show()

  def classesProbability(self,threshold,histogram,image_size):
      p1 = np.sum(histogram[0:threshold])/image_size
      p2 = 1-p1
      return p1, p2

  def classesMean(self,threshold, histogram):
      mu1 = 0
      mu2 = 0
      sum1_ = 0
      sum2_ = 0
      for i in range(threshold):
        sum1_ += i*histogram[i]
        mu1 = sum1_/np.sum(histogram[0:threshold])
      for i in range(threshold,6):
        sum2_ += i*histogram[i]
        mu2 = sum2_/np.sum(histogram[threshold:])
      return mu1, mu2

  def getThreshold(self):
      image = self.image
      histogram = self.getHistogram()
      image_size = self.image_size
      u = 1
      max_var = 0
      optimal_threshold = u
      while u < 256:
        p1, p2 = self.classesProbability(u,histogram,image_size)
        m1, m2 = self.classesMean(u,histogram)
        between_class_variance = p1*p2*((m1-m2)**2)
        if between_class_variance >  max_var:
          max_var = between_class_variance
          optimal_threshold = u
        u += 1
      return optimal_threshold

  def applyThresholding(self):
      image = self.image
      histogram = self.getHistogram()
      optimal_threshold = self.getThreshold()
      otsu_img = applyThreshold(image, optimal_threshold)
      return otsu_img

  def getOutput(self):
      output = self.applyThresholding()
      visualizePixels(output)