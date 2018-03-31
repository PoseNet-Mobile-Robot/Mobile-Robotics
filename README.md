# GTSAM with Posenet Factor

This is the GTSAM iSAM2 solver equipped with Posenet as 
a sensor model and odometry get from wheel encoder as an 
action model.

## Dependencies
1. [GTSAM 4.0](https://bitbucket.org/gtborg/gtsam/) with python2.7 enabled
2. python 2.7
3. numpy

## Attributes
1. add an odometry by calling solver.step(odometry, odometryNoise)
2. add a measurement by calling solver.addObs(measurement, measurementNoise)
3. update the graph and return the current estimate by calling solver.update()

