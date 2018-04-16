## main.py
## Author: Zongtai Luo
## Brief: run the posenet gtsam


from readOdometry import odometry
from readImage import readImage
import gen_data_nclt_new
from gtsamSolver import PoseNetiSam
from train import trainer
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pickle
import sys
import time
import os
import pdb


def main(pathOdom_,pathImage_,pathWeight_):

    # file path
    pathOdom = pathOdom_
    pathImage = pathImage_
    pathWeight = pathWeight_
    
    # initilize the gtsam solver
    iSam = PoseNetiSam()
    
    # initilize posenet
    poseNet = trainer(pathWeight,pathImage,100,False,False,True) # comment for debug

    # initialize the plot tool
    fig = plt.figure(1)
    
    # parameter
    resultsId = 1
    iSamUpdateRate = 1 # times of updates in one iteration
    timestamp = 0
    startId = 0
    
    # read data
    images = readImage(pathImage)
    datasource = gen_data_nclt_new.get_data()
    startTimestamp = images.getStartId() # start timestamp
    endTimestamp = images.getEndId() # end timestamp
    odom = odometry(pathOdom,startTimestamp,endTimestamp)
    
    # sensor info
    poseNetCov = [4.2,7.1,0.4]

    # records of path
    records = list()
    groundTruth = list()
    evalTime = list()
    
    # prior info
    priorMu = [32,-170,-0.3]
    priorCov = [10,10,10]
    iSam.initialize(priorMu,priorCov) # adding prior
    _,imgTimestamp,currGT = images.getImage(startId)
    image = datasource.images[startId]
    feed={tf.get_default_graph().get_tensor_by_name('Placeholder:0'): np.expand_dims(image, axis=0) }
    tic = time.time()
    measurement = poseNet.sess.run([tf.get_default_graph().get_tensor_by_name('fc9/fc9:0')], feed) # comment for debug
    measurement_x = np.squeeze(measurement)[0]
    measurement_y = np.squeeze(measurement)[1]
    measurement_theta = np.squeeze(measurement)[-1]
    measurement = np.array([measurement_x,measurement_y,measurement_theta])
    startId += 1
    iSam.addObs(measurement,poseNetCov) # adding first measurement # comment for debug
    # iSam.addObs(currGT,poseNetCov) # comment for release
    # iSam.printGraph()
    currEstimate = iSam.update() # update the graph
    evalTime.append(time.time()-tic)
    # currEstimate = priorMu # comment for release
    records.append(currEstimate)
    groundTruth.append(currGT)

    iterations = images.length()-1
    # iterations = 50
    # localization begins here
    print("\nBegin localization:\n")
    for i in tqdm(range(iterations)):

        _,imgTimestamp,currGT = images.getImage(startId)
        image = datasource.images[startId]        
        feed={tf.get_default_graph().get_tensor_by_name('Placeholder:0'): np.expand_dims(image, axis=0) }
       
        
        tic = time.time()
        # adding odometry
        # BETA: matching the frequency with the sensor measurement, cumsum the motion to reduce factor. Gaussian Assumption of the odometry and independence are made
        motionCum = [0,0,0]
        motionCovCum = [0,0,0]
        while timestamp<=imgTimestamp:
            motion,motionCov,timestamp = odom.getOdometry()
            motionCum = [motion[0]+motionCum[0],
                         motion[1]+motionCum[1],
                         motion[2]+motionCum[2]]
            motionCovCum = [motionCov[0]+motionCovCum[0],
                            motionCov[1]+motionCovCum[1],
                            motionCov[2]+motionCovCum[2]]
            # iSam.printGraph() # comment for release

        # print("ready to step.\n")
        iSam.step(motionCum,motionCovCum)

        # getting measurement and update image timestamp
        # measurement = poseNet.test(image,0,1) # comment for debug
        # measurement = [measurement[0],measurement[1],measurement[-1]]
        measurement = poseNet.sess.run([tf.get_default_graph().get_tensor_by_name('fc9/fc9:0')], feed) # comment for debug
        measurement_x = np.squeeze(measurement)[0]
        measurement_y = np.squeeze(measurement)[1]
        measurement_theta = np.squeeze(measurement)[-1]
        measurement = np.array([measurement_x,measurement_y,measurement_theta])
        groundTruth.append(currGT)

        # print("ready to add measurement.\n")
        # adding measurement factor
        iSam.addObs(measurement,poseNetCov) # comment for debug
        # iSam.addObs(currGT,poseNetCov) # comment for release
        
        # optimize factor graph
        # print("ready to update.\n")
        currentEst = iSam.update(iSamUpdateRate) # comment for debug
        evalTime.append(time.time()-tic)
        # currentEst = currGT # comment for release
        records.append(currentEst)
        
        # increment the Id
        startId += 1

        # plot tool for the calculated trajetory
        x = records[-1][0]
        y = records[-1][1]
        plt.scatter(x,y,c='g') # plot line from x_t-1 to x_t
        x = groundTruth[-1][0]
        y = groundTruth[-1][1]
        plt.scatter(x,y,c='r') # plot line of ground truth from x_t-1 to x_t
        plt.scatter(measurement[0],measurement[1],c='b') # plot points of posenet prediction
        if(i == 0):
            plt.legend(['prediction',' ground truth','posenet prediction'])

        # plt.draw() # comment if not dynamic plotting
        # plt.pause(0.01) # whether need to pause?
        # print("Done {} iterations, {} iterations in total".format(i+1,iterations))
        
    # store estimation
    pickleOut = open('dict.{0}_estimation'.format(resultsId),'wb')
    pickle.dump(np.asarray(records),pickleOut)
    pickleOut.close()

    # store ground truth
    pickleOut = open('dict.{0}_groundTruth'.format(resultsId),'wb')
    pickle.dump(np.asarray(groundTruth),pickleOut)
    pickleOut.close()    

    # store evalTime
    pickleOut = open('dict.{}_evalTime'.format(resultsId),'wb')
    pickle.dump(np.asarray(evalTime),pickleOut)
    pickleOut.close()
    
    # save fig
    fig.savefig("Trajectory{0}".format(resultsId))

if __name__=='__main__':

    # use gpu 0
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # reading argument
    argv = sys.argv
    if len(argv) < 3:
        argv = list()
        argv.append('/home/eecs568/Documents') # path to odometry
        argv.append('/home/eecs568/eecs568/Mobile-Robotics/cam4_test/') # path to images
        argv.append('/home/eecs568/eecs568/Mobile-Robotics/success_models/nclt_new/20180409-130922model_epoch_4.ckpt') # path to weights
    
    main(argv[0],argv[1],argv[2])
