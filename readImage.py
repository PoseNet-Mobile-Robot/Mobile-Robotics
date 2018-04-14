import os
import numpy as np
import cv2


class readImage():
    '''readImage is a class that reads the images from the path to folder
       and give images to the system by getImage()
    '''

    def __init__(self,pathImage):

        images = list()
        tail_str = '.tiff'
        for name in sorted(os.listdir(pathImage)):
            if name[-len(tail_str):] != tail_str:
                continue
            images.append(pathImage+name)

        self.imageLocs = images

        data = np.loadtxt(os.path.join(pathImage,'dataset_test.csv'),delimiter = ",")
        self.timestamp = data[:,0].astype(int)
        self.groundTruth = data[:,1:]
        self.currKey = 0
        
    def getImage(self,imageId,isPlot=False):
        image = cv2.imread(self.imageLocs[imageId],1)
        
        if isPlot:
            cv2.imshow('image{0}'.format(imageId),image)

        timestamp = self.timestamp[imageId]
        groundTruth = self.groundTruth[imageId,:]
        self.currKey += 1
        
        return image, timestamp, [groundTruth[0],groundTruth[1],groundTruth[5]]

    def getStartId(self):
        return self.timestamp[0]

    def getEndId(self):
        return self.timestamp[-1]

    def length(self):
        return len(self.timestamp)
