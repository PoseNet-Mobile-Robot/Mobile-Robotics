import numpy as np
import os


class odometry(object):
    '''
    odometry is a class that read and distribute odometries
    '''
    def __init__(self, filepath, startId=None, endId=None):

        mu_path = os.path.join(filepath,'odometry_mu.csv')
        cov_path = os.path.join(filepath,'odometry_cov.csv')

        odom = np.loadtxt(mu_path, delimiter=",")
        # reading the timestamps from the file for synchronize with sensor
        self.timestamps = odom[:,0]

        if startId == None:
            startId = 0
            endId = len(odom)

        # reading odometry mu (delta_x,delta_y,delta_theta)
        x = odom[startId:endId, 1]
        y = odom[startId:endId, 2]
        theta = odom[startId:endId, 6]
        self.odom = np.transpose(np.array([x,y,theta]))

        # reading the covariance matrix
        odomCov = np.loadtxt(cov_path, delimiter=",")

        self.odomCov = np.transpose(np.array([odomCov[startId:endId,1], odomCov[startId:endId,7], odomCov[startId:endId,21]]))

        # current reading index
        self.readingId = 0

    def getOdometry(self,readingId = None):
        # when getting new odometry the reading id will increment
        if readingId == None:
            readingId = self.readingId

        ## accumulate the odometry if reading is not continuous
        ## if the file is set to correspond with the image file then ommit this comment

        self.readingId += 1

        return list(self.odom[readingId,:]),list(self.odomCov[readingId,:]),self.timestamps[readingId]

    def printOdometry(self,printingId = None):
        if printingId == None:
            printingId = self.readingId+1

        print "The next odometry is:",list(self.odom[printingId,:]),"\nThe Covariance is",list(self.odomCov[printingId,:])
