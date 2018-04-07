import os
import numpy as np
import cv2


class readImage():
    '''readImage is a class that reads the images from the path to folder
       and give images to the system by getImage()
    '''

    def __init__(self,pathImage):
        imagesLocs = [os.path.join(root, name) for root, dirs, files in os.walk(pathImage)
                  for name in files if name.endswith((".png",".jpg",".jpeg",".gif"))]
        self.imageLocs = images

        data = nploadtxt(os.join(pathImage,'timestamp.txt'),delimiter = ",")
        self.timestamp = data[:,0]
        self.groundTruth = data[:,1:]
        self.currKey = 0
        
    def getImage(self,imageId,isPlot=False):
        image = cv2.imread(self.imageLocs[imageId],1)
        
        if isPlot:
            cv2.imshow('image{0}'.format(imageId),image)

        key = self.timestamp[self.currKey]
        groundTruth = self.groundTruth[self.currKey,:]
        self.currKey += 1
        
        return image, key, [groundTruth[0],groundTruth[1],groundTruth[5]]
