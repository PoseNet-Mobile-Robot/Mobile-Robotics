# Deep Learning in SLAM
Application of PoseNet and dynamic structural data generation for real-time localization

## Dependencies
1. [IP WebCam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en) 
2. Python 2.7
3. OpenCV 3
4. TensorFlow
5. MatPlotlib/ NumPy/ urllib2
6. [GTSAM 4.0](https://bitbucket.org/gtborg/gtsam/) with Python 2.7 enabled

## Procedure
1. Install IP WebCam on your phone
2. Replace variable "url" in VoIP.py with hosted video url

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
