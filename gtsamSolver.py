from __future__ import print_function
import gtsam
import numpy as np


# a hash-like function
X = lambda i: int(gtsam.symbol(ord('x'), i))


class PoseNetiSam(object):
    """A solver for posenet specified SLAM

    """
    def __init__(self, relinearizeThreshold=0.01, relinearizeSkip=1):
        """ priorMean and priorCov should be in 1 dimensional array
        """

        # init the graph
        self.graph = gtsam.NonlinearFactorGraph()

        # init the iSam2 solver
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(relinearizeThreshold)
        parameters.setRelinearizeSkip(relinearizeSkip)
        self.isam = gtsam.ISAM2(parameters)

        # init container for initial values
        self.initialValues = gtsam.Values()

        # setting the current position
        self.currentKey = 1

        # current estimate
        self.currentEst = False
        self.currentPose = [0,0,0]

        return

    def _motion_model(self, odometry):
        currPos = self.currentPose
        predPos = [currPos[0]+odometry[0], currPos[1]+odometry[1], currPos[2]+odometry[2]]
        return predPos

    def initialize(self, priorMean=[0,0,0], priorCov=[0,0,0]):
        # init the prior
        priorMean = gtsam.Pose2(priorMean[0], priorMean[1], priorMean[2])
        priorCov = gtsam.noiseModel_Diagonal.Sigmas(np.array(priorCov))
        self.graph.add(gtsam.PriorFactorPose2(X(self.currentKey), priorMean, priorCov))
        self.initialValues.insert(X(self.currentKey), priorMean)

        return

    def step(self, odometry, odometryNoise):
        odometryGT = gtsam.Pose2(odometry[0],odometry[1],odometry[2])
        odometryNoise = gtsam.noiseModel_Diagonal.Variances(np.array(odometryNoise))
        self.graph.add(gtsam.BetweenFactorPose2(X(self.currentKey), X(self.currentKey+1),
                                                odometryGT, odometryNoise))

        # adding the initialValues
        # TODO: when step function is called keep track of the current value instead          of calling the current position in motion model.
        predMean = self._motion_model(odometry)
        initialVal = gtsam.Pose2(predMean[0],predMean[1],predMean[2])
        self.initialValues.insert(X(self.currentKey+1), initialVal)

        # increment the key
        self.currentKey += 1
        # update current pose if adding odometry
        self.currentPose = predMean

        return

    def addObs(self, measurement, measurementNoise):
        measurement = gtsam.Pose2(float(measurement[0]),float(measurement[1]),float(measurement[2]))
        measurementNoise = gtsam.noiseModel_Diagonal.Variances(np.array(measurementNoise))
        self.graph.add(gtsam.PriorFactorPose2(X(self.currentKey), measurement, measurementNoise))

        return

    def update(self, updateNum = 1):
        self.isam.update(self.graph, self.initialValues)
        updateNum -= 1
        
        while updateNum > 0:
            self.isam.update()
            updateNum -= 1

        # clear graph and initial values
        self.graph.resize(0)
        self.initialValues.clear()
        self.currentEst = self.isam.calculateEstimate()
        # print(self.currentKey)
        # self.currentEst.print("\nEstimate:\n")
        # print(self.currentEst.atPose2(X(1)).x())
        '''print([self.currentEst.atPose2(X(self.currentKey)).x(),
               self.currentEst.atPose2(X(self.currentKey)).y(),
               self.currentEst.atPose2(X(self.currentKey)).theta()])
        print("Hello")
        '''
        # update current pose if update
        self.currentPose = [self.currentEst.atPose2(X(self.currentKey)).x(),
                            self.currentEst.atPose2(X(self.currentKey)).y(),
                            self.currentEst.atPose2(X(self.currentKey)).theta()]
        return self.currentPose

#    def currentPos(self, key=0):
#        if key == 0:
#            key = self.currentKey

#        if(self.currentEst):
#            currentPos = [self.currentEst.atPose2(X(key)).x(),
#                          self.currentEst.atPose2(X(key)).y(),
#                          self.currentEst.atPose2(X(key)).theta()]
#        else:
#            currentPos = [0,0,0]
#        return currentPos

    def printGraph(self, output = "\nFactor Graph:\n"):
        print(self.graph)
        return
        
    def printResult(self, output = "\nEstimation for vertices:\n"):

        print(output)

        '''for i in range(self.currentKey):
            pos = [self.currentEst.atPose2(X(self.currentKey)).x(),
                   self.currentEst.atPose2(X(self.currentKey)).y(),
                   self.currentEst.atPose2(X(self.currentKey)).theta()]
            print("x",str(i+1),":", pos, "\n")
        '''
        print(self.currentEst)
        return
