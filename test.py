import data_handler as DH
import numpy as np

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

location = "./ShopFacade/"

dh = DH.Process(location)

# pick number of images to pick samples from
numImages = 50
dh.generateData(numImages)
dh.remimages()

dh.numsamples()
flag, images, labels = dh.fetch(60)
print(images.shape, len(labels), flag)
dh.remsamples()

# Trick to pick samples from selected images
numSamples = 60
flag = True
while flag==True:
    flag, images, labels = dh.fetch(numSamples)
    dh.remsamples()
