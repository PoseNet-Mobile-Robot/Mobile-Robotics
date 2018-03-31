from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gtsamSolver


def main():
    # define some prior
    delta = 0.1
    priorMean = [delta, -delta, delta/10]
    priorCov = [delta/10, -delta/10, delta]

    # define odometry
    odometrys = [[5,0,0],[5,0,0],[5,0,-np.pi/2],[5,0,-np.pi/2],[5,0,-np.pi/2]]
    odoNoise = [0.5,0.5,1]

    # define measurements
    measurements = [[0,0,0],[5+2*delta,0+delta,0+delta/10],[10-4*delta,0-delta,-np.pi/2+delta/5],
                    [10+3*delta,-5+2*delta, -np.pi+delta],[5-delta,-5+delta,np.pi-delta/5],
                    [5-2*delta,0+3*delta,0-delta]]
    measureNoise = [0.1,0.1,0.1]

    solver = gtsamSolver.PoseNetiSam()
    solver.initialize(priorMean,priorCov)
    solver.addObs(measurements[0],measureNoise)
    solver.update()
    iterations = 5

    # Slam starts
    for i in range(iterations):
        solver.step(odometrys[i],odoNoise)
        solver.addObs(measurements[i+1], measureNoise)
        solver.update()

    solver.printResult()

if __name__ == '__main__':
    main()