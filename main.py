from readOdometry import odometry
from readImage import readImage
from gtsamSolver import PoseNetiSam
from train import trainer
import matplotlib.pyplot as plt
import pickle
import sys


def main(pathOdom_,pathImage_,pathWeight_):

    # file path
    pathOdom = pathOdom_
    pathImage = pathImage_
    pathWeight = pathWeight_
    
    # initilize the gtsam solver
    iSam = PoseNetiSam()
    
    # initilize posenet
    poseNet = trainer(pathWeight,pathImage) # comment for debug

    # initialize the plot tool
    # plt.ion() # comment for debug
    # fig,ax = plt.subplots() # comment for debug
    fig = plt.figure(1)
    
    # parameter
    resultsId = 1
    iSamUpdateRate = 1 # times of updates in one iteration
    timestamp = 0
    startId = 0
    
    # read data
    images = readImage(pathImage)
    startTimestamp = images.getStartId() # start timestamp
    endTimestamp = images.getEndId() # end timestamp
    odom = odometry(pathOdom,startTimestamp,endTimestamp)
    
    # sensor info
    poseNetCov = [0.5,0.5,0.5]

    # records of path
    records = list()
    groundTruth = list()
    
    # prior info
    priorMu = [32,-541,-2.5]
    priorCov = [0.5,0.5,0.5]
    iSam.initialize(priorMu,priorCov) # adding prior
    img,imgTimestamp,currGT = images.getImage(startId)
    measurement = poseNet.test(images.getImage(startId)) # comment for debug
    startId += 1
    iSam.addObs(measurement,poseNetCov) # adding first measurement # comment for debug
    # iSam.addObs(currGT,poseNetCov) # comment for release
    currEstimate = iSam.update() # update the graph
    # currEstimate = priorMu # comment for release
    records.append(currEstimate)
    groundTruth.append(currGT)

    iterations = images.length()-1

    # localization begins here
    for i in range(iterations):

        image,imgTimestamp,currGT = images.getImage(startId)
       
        # adding odometry
        # BETA: matching the frequency with the sensor measurement, for loop is needed
        while timestamp<=imgTimestamp:
            motion,motionCov,timestamp = odom.getOdometry()
            iSam.step(motion,motionCov)

        # getting measurement and update image timestamp
        measurement = poseNet.test(image) # comment for debug
        groundTruth.append(currGT)

        # adding measurement factor
        iSam.addObs(measurement,poseNetCov) # comment for debug
        # iSam.addObs(currGT,poseNetCov) # comment for release
        
        # optimize factor graph
        currentEst = iSam.update(iSamUpdateRate) # comment for debug
        # currentEst = currGT # comment for release
        records.append(currentEst)
        
        # increment the Id
        startId += 1

        # plot tool for the calculated trajetory
        # ax.plot(records[-2:,1],records[-2:,2],c='g') # plot line from x_t-1 to x_t
        # ax.plot(groundTruth[-2:,1],groundTruth[-2:,2],c='r') # plot line of ground truth from x_t-1 to x_t
        x = [records[-2][0],records[-1][0]]
        y = [records[-2][1],records[-1][1]]
        plt.scatter(x,y,c='g') # plot line from x_t-1 to x_t
        x = [groundTruth[-2][0],groundTruth[-1][0]]
        y = [groundTruth[-2][1],groundTruth[-1][1]]
        plt.scatter(x,y,c='r') # plot line of ground truth from x_t-1 to x_t
        if(i == 0):
            plt.legend(['prediction',' ground truth'])
        plt.pause(0.01) # whether need to pause?
        print("In {} iterations, {} of iterations in total".format(i,iterations))
        
    # store data
    pickleOut = open('dict.{0}_result'.format(resultsId),'wb')
    pickle.dump(records,pickleOut)
    pickleOut.close()

    # save fig
    fig.savefig("Trajectory{0}".format(resultsId))

if __name__=='__main__':

    # reading argument
    argv = sys.argv
    if len(argv) < 3:
        argv = list()
        argv.append('/home/eecs568/Documents') # path to odometry
        argv.append('/home/eecs568/Documents/TestImages') # path to images
        argv.append('/home/eecs568/Mobile-Robotics/success_models/ShopFacade_weights') # path to weights
    
    main(argv[0],argv[1],argv[2])
