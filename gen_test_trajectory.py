import tensorflow as tf
import os, sys
import gen_data
import train
import numpy as np
import pdb
import math, transforms3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# second to last argument False for nclt True for others
#weight_path = 'success_models/ShopFacade_weights/model_epoch_3.ckpt'
#image_path = '.ShopFacade/'
#fig_name = 'ShopFacade Trajectory.png'

weight_path = 'success_models/KingsCollege/model_epoch_90.ckpt'
image_path = './KingsCollege/'
fig_name = 'KingsCollege Trajectory.png'
trainer = train.trainer(weight_path, image_path, True, True, True)
datasource = gen_data.get_data()

# initialize plot tool
fig = plt.figure(1)

error = np.zeros([len(datasource.images),3])

for i in range(len(datasource.images)):
    np_image = datasource.images[i]
    feed={tf.get_default_graph().get_tensor_by_name('Placeholder:0'): np.expand_dims(np_image, axis=0) }

    # ground truth x y z
    pose_x= np.asarray(datasource.poses[i][0:3])

    # ground truth euler angles
    pose_q= np.asarray(datasource.poses[i][3:7]) 
    pose_euler_angle = transforms3d.euler.quat2euler(pose_q)
    
    x_q = trainer.sess.run([tf.get_default_graph().get_tensor_by_name('fc9/fc9:0') ], feed)

    # x y z
    pred_x = np.squeeze(x_q)[0:3]

    # euler angle
    pred_q = np.squeeze(x_q)[3:7]
    pred_euler_angle = transforms3d.euler.quat2euler(pred_q)

    # scatter plot for pose
    plt.scatter(pose_x[0],pose_x[1],c='g')
    plt.scatter(pred_x[0],pred_x[1],c='r')

    error[i,:] = np.array([pose_x[0]-pred_x[0],pose_x[1]-pred_x[1],pose_q[-1]-pred_q[-1]])
    
# save the plot
plt.legend(['ground truth','prediction'])
fig.savefig(fig_name)

# calculate stddev and mean error
meanErr = np.sum(error,axis=0)/len(error)
stdErr = np.std(error,axis=0)
print("The mean error is {} and standard deviation is {}.".format(meanErr,stdErr))
