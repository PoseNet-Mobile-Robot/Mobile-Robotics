import cv2
import sys, os
from random import randint, sample
import numpy as np
import pdb
class Process:
    '''
    functions:
    OSWalk() = walks through directory and lists all the images in the directory
    generateData(num) = selects a random set of 'num' images from the listed images
    process(num, indices) = converts the selected 'num' images to 128*num samples
    fetch(num) = selects a random set of 'num' samples from the produced samples using process()

    helper functions:
    getLabels() = walks through directory and populates the labels
    getName(name) = converts ./path/to/here/img.png to here/img.png
    reset() = resets the counters when you have used up all the images
    numimages() = prints total number of images in the directory
    numsamples() = prints total number of samples generated from selected images
    remimages() = prints total number of images that remain in the directory
    remsamples() = prints total number of samples that remain from the selected lot of images
    '''

    def __init__(self, location, height = 224, width = 224, depth = 3):
        self.img2labels = dict()
        self.idx2img = dict()
        self.num_crops=1
        self.location = location
        self.height = height
        self.width = width
        self.depth = depth
        self.numImages = 0
        self.numSamples = 0
        self.imageLocs = []

        self.sampleImages = []
        self.remImages = []
        self.remSamples = []

        # OS Walk for getting all image files
        self.OSWalk()
        # print number of images
        self.numimages()
        # populate labels dict
        self.getLabels()


    def OSWalk(self):
        images = [os.path.join(root, name) for root, dirs, files in os.walk(self.location)
             for name in files if name.endswith((".png", ".jpg", ".jpeg", ".gif"))]
        self.imageLocs = images

        # for i in range(len(images)):
        #    a,b = images[i].split('\\')
        #    self.imageLocs[i] = a + '/' + b

        # array to store which image has been used
        self.numImages = len(self.imageLocs)
        self.remImages = np.ones((self.numImages), dtype = bool)



    def generateData(self, num):
        ctr = 0
        indices = ['']*num
        rem = np.sum(self.remImages)

        # Pick random images
        if rem < num:
            print('Number of Images left to select: ', rem)
            print('Generating samples for the remaining images')

        num = min(rem, num) # only activated when the number of images left is less than the request

        while ctr<num:
            idx = randint(0,self.numImages-1)
            if self.remImages[idx]==True:
                self.remImages[idx] = False
                indices[ctr] = self.imageLocs[idx]
                ctr += 1

        # array to store which sample has been used
        self.numSamples = self.num_crops*num
        self.remSamples = np.ones((self.numSamples), dtype = bool)

        # generate samples
        self.process(num, indices)

    def centeredCrop(self, img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	cropped_img = img[height_offset:height_offset + output_side_length,
						width_offset:width_offset + output_side_length]
	return cropped_img


    def process(self, num, indices):
        print('Generating Samples .... ')

        self.sampleImages = np.zeros((num*self.num_crops , self.height, self.width, self.depth), dtype=np.uint8)

        idx = 0
        imgs = np.zeros((num,  256, 455, 3))
        names = []
        for i in range(num):
            # read image
            img = cv2.resize(cv2.imread(indices[i], 1).astype(float), (455,256), interpolation = cv2.INTER_CUBIC)
            imgs[i] = img
            name = self.getName(indices[i])
            names.append(name)
        means = np.mean(imgs, axis=0)
        imgs = imgs - means
        #temp_mean = np.mean(img, axis=0)
        #temp_mean = np.mean(temp_mean, axis=0)
        #temp_std = np.zeros(self.depth)

        #for i in range(self.depth):
        #    img[:,:,i] -= temp_mean[i]
        #    temp_std[i] = np.std(img[:,:,i])
        #    img[:,:,i] /= temp_std[i]
                
        for i in range(num):
            # generate 128 random indices for crop
            for j in range(idx, idx+self.num_crops ):
                x = randint(0,31)
                y = randint(0,230)
                img = imgs[i, :,:,:]
                self.sampleImages[j, :, :, :] = self.centeredCrop(img, 224)#img[x:x + self.height, y:y + self.width, :].copy()
                self.idx2img[j] = names[i]
            idx += self.num_crops



    def fetch(self, num):
        samples = np.zeros((num, self.height, self.width, self.depth), dtype=np.float)
        labels = ['']*num
        ctr = 0
        flag = True
        # number of remaining samples
        rem = np.sum(self.remSamples)

        if rem<num:
            flag = False
            print("Number of samples left to select: ", rem)

        else:
            while ctr<num:
                idx = randint(0,self.numSamples-1)
                if self.remSamples[idx]==True:
                    self.remSamples[idx] = False

                    # gaussian normalization of image to have mean 0, variance 1
                    temp = self.sampleImages[idx, :, :, :].astype(float)
                    try:
                        labels[ctr] = self.img2labels[self.idx2img[idx]]
                    except KeyError:
                        pdb.set_trace()
                    ctr += 1

        return [flag, samples, labels]



    def getLabels(self):
        f1 = open(self.location+'dataset_test.txt')
        lines = f1.readlines()
        f2 = open(self.location+'dataset_train.txt')
        lines2 = f2.readlines()

        # append the total list and process labels
        lines.extend(lines2)
        numLines = len(lines)

        # we assume that the dataset_train.txt has no redundant texts at the beginning
        for i in range(numLines):
            line = lines[i].strip()
            line = line.split()
            self.img2labels[line[0]] = list(map(float,line[1:8]))
            
    def getName(self, loc):
        ctr = 0
        for i in range(len(loc)):
            if loc[i]=='/':
                ctr += 1
            if ctr==2:
                return loc[i+1:]



    def reset(self):
        self.remImages = np.ones((self.numImages), dtype = bool)



    def numimages(self):
        print('The total number of images are: ', self.numImages)
        return self.numImages



    def numsamples(self):
        print('The total number of samples are: ', self.numSamples)
        return self.numSamples
        

    def remsamples(self):
        print('The number of samples that remain are: ',np.sum(self.remSamples))
        return self.remSamples


    def remimages(self):
        print('The number of images that remain are: ',np.sum(self.remImages))
        return remImages
