import cv2
import sys, os
from random import randint, sample
import numpy as np

class preprocess:
    def __init__(self, location, height = 224, width = 224, depth = 3):
        self.labels = dict()    # dictionary between labels and quaternions
        self.maps = dict()   # dicitonary between labels and ID
        self.location = location + 'seq1/'
        self.dirElem = os.listdir(self.location)
        self.lenDir = len(self.dirElem)
        self.height = height
        self.width = width
        self.depth = depth
        self.numsamples = 128*self.lenDir
        self.sampleImages = np.zeros((self.height, self.width, self.depth*self.numsamples), dtype=np.uint8)
        self.ret = np.ones((self.numsamples), dtype = bool)

        # read all images in directory and store in np array
        self.process()

        # store all labels in a dictionary
        self.getLabels(location)

    def reset(self):
        self.sampleImages = np.zeros((self.height, self.width, self.depth*self.numsamples), dtype=np.uint8)
        self.ret = np.ones((self.numsamples), dtype = bool)
        # read all images in directory and store in np array
        self.process()
        # store all labels in a dictionary
        self.getLabels(location)


        
    def numSamples(self):
        return self.numsamples

    def samplesLeft(self):
        return np.sum(self.ret)

    def fetch(self, num):
        # return random sample of size
        return self.randomSample(num)

    # resize images to 455 x 256 from 1920 x 1080 and picks 128 random 224 x 224 samples from each
    def process(self):
        print('Directory: '+self.location)
        print('Computing Samples...')
        elems = ['']*self.lenDir
        idx = 0
        for i in xrange(self.lenDir):
            elems[i] = self.location + self.dirElem[i]
            img = cv2.resize(cv2.imread(elems[i],1), (455,256), interpolation=cv2.INTER_CUBIC)
            [h, w, d] = img.shape   # (256, 455, 3)

            # height random crop range = 0-31
            # width random crop range = 0-230
            for j in xrange(128):
                self.maps[idx] = self.dirElem[i]
                x = randint(0,31)
                y = randint(0,230)
                self.sampleImages[:, :, idx:idx+3] = img[x:x+self.height, y:y+self.width,:].copy()
                idx += 3

    def getLabels(self, location):
        print('Processing Labels...')

        f = open(location+'dataset_test.txt')
        lines = f.readlines()
        numLines = len(lines)

        for i in xrange(3,numLines):
            line = lines[i].strip()
            line = line.split()
            # extract just the name of the file
            name = line[0][5:]
            self.labels[name] = list(map(float,line[1:8]))

    def randomSample(self, num):
        i = 0
        out = np.zeros((num, self.height, self.width, self.depth), dtype=np.float)
        labels = [[] for x in xrange(num)]

        if np.sum(self.ret) < num:
            raise 'all samples of dataset used'
            exit(0)
        else:
            while i<num:
                idx = randint(0,self.numsamples-1)
                if self.ret[idx]==True:
                    out[i,:,:,:] = self.sampleImages[:,:, idx:idx+3]
                    self.ret[idx] = False
                    labels[i] = self.labels[self.maps[self.depth*idx]]
                    i += 1
                    # print('Sample: ',i, '  |  Label: ', self.maps[self.depth*idx], '  |  Attributes: ', self.labels[self.maps[self.depth*idx]])

        # normalize output to 0-1
        out *= 1/out.max()
        return out, np.array(labels)

    def reset(self):
        self.ret = np.ones((self.numsamples), dtype = bool)
