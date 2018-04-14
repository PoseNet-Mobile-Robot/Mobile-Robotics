import tensorflow as tf
import os, sys
import gen_data_nclt_new
import train
import numpy as np
import pdb
import math, transforms3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# second to last argument False for nclt True for others
weightPath = '/home/eecs568/eecs568/Mobile-Robotics/success_models/nclt_new/20180409-130922model_epoch_4.ckpt'
#imagePath = './cam4_train/'
#figname = 'nclt_train.png'
imagePath = '/home/eecs568/Documents/TestImages\ 2012-01-08/test/'
figname = 'nclt_test_seq2.png'
# trainer = train.trainer(weightPath, imagePath, 100, False, False)
datasource = gen_data_nclt_new.get_data()

# initialize plot tool
fig = plt.figure(1)

error = np.zeros([len(datasource.images),3])

for i in range(len(datasource.images)):
    np_image = datasource.images[i]
    # feed={tf.get_default_graph().get_tensor_by_name('Placeholder:0'): np.expand_dims(np_image, axis=0) }

    # ground truth x y z
    pose_x= np.asarray(datasource.poses[i][0:2])

    # ground truth euler angles
    pose_q= np.asarray(datasource.poses[i][3:6]) 
    # pose_euler_angle = transforms3d.euler.quat2euler(pose_q)
    
    # x_q = trainer.sess.run([tf.get_default_graph().get_tensor_by_name('fc9/fc9:0') ], feed)
    # pdb.set_trace()

    # x y z
    # pred_x = np.squeeze(x_q)[0:3]

    # euler angle
    # pred_q = np.squeeze(x_q)[3:6]
    # pred_euler_angle = transforms3d.euler.quat2euler(pred_q)

    # scatter plot for pose
    plt.scatter(pose_x[0],pose_x[1],c='g')
    # plt.scatter(pred_x[0],pred_x[1],c='r')
    # plt.pause(0.01)
    # plt.draw()
    # error[i,:] = np.array([pose_x[0]-pred_x[0],pose_x[1]-pred_x[1],pose_q[-1]-pred_q[-1]])
    
    print("iteration {}\n".format(i))
    
# save the plot
plt.legend(['ground truth','prediction'])
fig.savefig(figname)

# calculate stddev and mean error
#meanErr = np.sum(error,axis=0)/len(error)
#stdErr = np.std(error,axis=0)
#print("The mean error is {} and standard deviation is {}.".format(meanErr,stdErr))

