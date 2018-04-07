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
    poseNet = trainer(pathWeight,pathImage)

    # initialize the plot tool
    plt.ion()
    fig,ax = plt.subplots()
    
    # parameter
    resultsId = 1
    iSamUpdateRate = 1 # times of updates in one iteration
    timestamp = 0
    
    # read data
    images = readImage(pathImage)
    startId = images.getStartId()
    odom = odometry(pathOdom,startId)
    # TODO: read the ground truth for plot comparision
    
    # sensor info
    poseNetCov = [0.5,0.5,0.5]

    # records of path
    records = list()
    groundTruth = list()
    
    # prior info
    priorMu = [0,0,0]
    priorCov = [0,0,0]
    iSam.initialize(priorMu,priorCov) # adding prior
    img,imgTimestamp,currGT = images.getImage(startId)
    measurement = poseNet.test(images.getImage(startId))
    startId += 1
    iSam.addObs(measurement,poseNetCov) # adding first measurement
    currEstimate = iSam.update(2) # update the graph
    records.append(currEstimate)
    groundTruth.append(currGT)

    iterations = images.length()
    
    # localization begins here
    for i in range(iterations):

        # adding odometry
        # BETA: matching the frequency with the sensor measurement, for loop is needed
        while timestamp<=imgTimestamp:
            motion,motionCov,timestamp = odom.getOdometry()
            iSam.step(motion,motionCov)

        # getting measurement and update image timestamp
        measurement,imgTimestamp,currGT = poseNet.test(images.getImage(startId))
        groundTruth.append(currGT)

        # adding measurement factor
        iSam.addObs(measurement,poseNetCov)

        # optimize factor graph
        currentEst = iSam.update(iSamUpdateRate)
        records.append(currentEst)
        
        # increment the Id
        startId += 1

        # plot tool for the calculated trajetory
        ax.plot(records[-2:,1],records[-2:,2],c='g') # plot line from x_t-1 to x_t
        ax.plot(groundTruth[-2:,1],groundTruth[-2:,2],c='r') # plot line of ground truth from x_t-1 to x_t
        plt.pause(0.01) # whether need to pause?
        plt.draw()
        
    # store data
    pickleOut = open('dict.{0}_result'.format(resultsId),'wb')
    pickle.dump(records,pickleOut)
    pickleOut.close()

    # save fig
    fig.save("Trajectory{0}".format(resultsId))

if __name__=='__main__':

    # reading argument
    argv = sys.argv
    if argv < 3:
        argv = list()
        argv.append('') # path to odometry
        argv.append('') # path to images
        argv.append('') # path to weights
    
    main(argv[0],argv[1],argv[2])
