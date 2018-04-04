from readOdometry import odometry
from readImage import readImage
from gtsamSolver import PoseNetiSam
from train import trainer
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

    # parameter
    resultsId = 1
    iSamUpdateRate = 2 # times of updates in one iteration
    startId = 0 # start image id
    endId = 1e3 # end image id
    numInterval = 1
    iterations = int((endId-startId)/numInterval)
    
    # read data
    odom = odometry(pathOdom,startId,endId)
    images = readImage(pathImage)
    #TODO: read the ground truth for plot comparision
    
    # sensor info
    poseNetCov = [0.5,0.5,0.5]

    # records of path
    records = list()
    
    # prior info
    priorMu = [0,0,0]
    priorCov = [0,0,0]
    iSam.initialize(priorMu,priorCov) # adding prior
    measurement = poseNet.test(images.getImage(startId))
    startId += numInterval
    iSam.addObs(measurement,poseNetCov) # adding first measurement
    iSam.update(2) # update the graph
    
    # localization begins here
    for i in range(iterations):

        # adding odometry
        iSam.step(odom.getOdometry(startId))

        # getting measurement
        measurement = poseNet.test(images.getImage(startId))

        # adding measurement factor
        iSam.addObs(measurement,poseNetCov)

        # optimize factor graph
        currentEst = iSam.update(iSamUpdateRate)
        records.append(currentEst)
        
        # increment the Id
        startId += numInterval

        # TODO: plot tool for the calculated trajetory
      
        
    # store data
    pickleOut = open('dict.{0}_result'.format(resultsId),'wb')
    pickle.dump(records,pickleOut)
    pickleOut.close()
    

if __name__=='__main__':

    # reading argument
    argv = sys.argv
    if argv < 3:
        argv = list()
        argv.append('') # path to odometry
        argv.append('') # path to images
        argv.append('') # path to weights
    
    main(argv[0],argv[1],argv[2])
