import data_handler as DH
import numpy as np
import cv2
'''
Usage:
location : enter location of main directory
generateData(num) = selects a random set of 'num' images from the listed images
fetch(num) = selects a random set of 'num' samples from the produced samples using process()
reset() = resets the counters when you have used up all the images
numimages() = prints total number of images in the directory
numsamples() = prints total number of samples generated from selected images
remimages() = prints total number of images that remain in the directory
remsamples() = prints total number of samples that remain from the selected lot of images
'''

# location = "./2013-01-10/"
# file = 'groundtruth_2013-01-10.csv'

location = "/home/eecs568/eecs568/Mobile-Robotics/nclt/"
file = 'groundtruth_2012-03-17.csv'

dh = DH.Process(location, file, True)

# pick number of images to pick samples from
numImages = 1000
dh.generateData(numImages)
dh.remimages()

dh.numsamples()
flag, images, labels = dh.fetch(6)
print(images.shape, len(labels), flag)
dh.remsamples()

# Trick to pick samples from selected images
numSamples = 10
flag = True
while flag==True:
    flag, images, labels = dh.fetch(numSamples)
    dh.remsamples()

