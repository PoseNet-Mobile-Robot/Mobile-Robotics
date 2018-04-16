import pdb
from tqdm import tqdm
import cv2, os, sys
import numpy as np
import random, imutils
batch_size = 32
# train directory
#directory = './nclt_tripple/'
#dataset = 'dataset_train.csv'

# test directory
directory = './nclt_03_31/test/'
dataset = 'dataset_test.csv'

class datasource(object):
    def __init__(self, images, poses):
        print("Image Data path: "+directory)
        print("label path: "+dataset)
        self.images = images
        self.poses = poses

def centeredCrop(img, output_side_length):
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

def preprocess(images):
    images_out = [] #final result
    #Resize and crop and compute mean!
    images_cropped = []
    for i in tqdm(range(len(images))):
        X = cv2.imread(images[i])
        if X.shape[0] < X.shape[1]:
            X = imutils.resize(X , height=256)
        else:
            X = imutils.resize(X, width=256)
        X = imutils.rotate(X, angle=270)
        #X = cv2.resize(X, (455, 256))
        X = centeredCrop(X, 224)
        images_cropped.append(X)
    #compute images mean
    N = 0
    mean = np.zeros((1, 3, 224, 224))
    for X in tqdm(images_cropped):
        mean[0][0] += X[:,:,0]
        mean[0][1] += X[:,:,1]
        mean[0][2] += X[:,:,2]
        N += 1
    mean[0] /= N
    #Subtract mean from all images
    for X in tqdm(images_cropped):
        X = np.transpose(X,(2,0,1))
        X = X - mean
        X = np.squeeze(X)
        X = np.transpose(X, (1,2,0))
        images_out.append(X)
    return images_out

def get_data():
    poses = []
    images = []
    line_num = 0
    all_imgs = sorted(os.listdir(directory))[:-1]
    with open(directory+dataset) as f:
        #next(f)  # skip the 3 header lines
        #next(f)
        #next(f)
        for line in f:
            fname, p0,p1,p2,p3,p4,p5 = line.split(',')
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            try:
                filename = directory+'/'+all_imgs[line_num] #fname+".tiff"
            except:
                pdb.set_trace()
            if (os.path.isfile(filename)==False):
                pdb.set_trace()
                continue
            else:
                poses.append((p0,p1,p2,p3,p4,p5))
                images.append(filename )
            line_num += 1
    print("Num of images is "+str(len(images)))
    images = preprocess(images)
    return datasource(images, poses)

def gen_data(source):
    while True:
        indices = range(len(source.images))
        random.shuffle(indices)
        for i in indices:
            image = source.images[i]
            pose_x = source.poses[i][0:3]
            pose_q = source.poses[i][3:]
            yield image, pose_x, pose_q

def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        pose_q_batch = []
        for _ in range(batch_size):
            image, pose_x, pose_q = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
            pose_q_batch.append(pose_q)
        yield np.array(image_batch), np.array(pose_x_batch), np.array(pose_q_batch)
