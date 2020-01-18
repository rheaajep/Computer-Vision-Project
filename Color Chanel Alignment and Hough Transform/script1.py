import numpy as np 
import cv2
from alignChannels import alignChannels
# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red=np.load('/home/geekerlink/Desktop/Computer Vision/hw1/hw1/data/red.npy')
green=np.load('/home/geekerlink/Desktop/Computer Vision/hw1/hw1/data/green.npy')
blue=np.load('/home/geekerlink/Desktop/Computer Vision/hw1/hw1/data/blue.npy')

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
cv2.imshow("rgbResult",rgbResult)
cv2.waitKey(0)
cv2.imwrite('/home/geekerlink/Desktop/Computer Vision/hw1/hw1/results/rgb_output.jpg',rgbResult)