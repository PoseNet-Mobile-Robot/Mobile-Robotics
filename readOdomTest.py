from readOdometry import odometry

# change this to the odometry folder
filepath = 'path/to/odometry'

# load odometry and print the first item
odom = odometry(filepath)
odom.printOdometry()

# get the next odometry and increment the timestamp
print odom.getOdometry()
