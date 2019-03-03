# Deep Learning in SLAM
Application of PoseNet and dynamic structural data generation for real-time localization

## Dependencies
1. Python 2.7
2. OpenCV 3
3. TensorFlow
4. MatPlotlib/ NumPy/ urllib2
5. [GTSAM 4.0](https://bitbucket.org/gtborg/gtsam/) with Python 2.7 enabled

# GTSAM with Posenet Factor
This is the GTSAM iSAM2 solver equipped with Posenet as
a sensor model and odometry get from wheel encoder as an
action model.

## Attributes
1. add an odometry by calling solver.step(odometry, odometryNoise)
2. add a measurement by calling solver.addObs(measurement, measurementNoise)
3. update the graph and return the current estimate by calling solver.update()

## References
1. [PoseNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf)
2. [Structural Data Generation](http://ccwu.me/vsfm/vsfm.pdf)
3. [Georgia Tech Smoothing and Mapping (GTSAM)](https://borg.cc.gatech.edu/)
4. [NCLT Dataset](http://robots.engin.umich.edu/nclt/)
