import tensorflow as tf
import os, sys
import gen_data_cam
import train
import numpy as np
import pdb
import math, transforms3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# second to last argument False for nclt True for others
weightPath = '20180415-201632model_epoch_4.ckpt'
imagePath = './nclt_03_31/test/'
#figname = 'nclt_train.png'
#imagePath = './nclt_11_04/test/'
figname = 'nclt_train_four.png'
trainer = train.trainer(weightPath, imagePath, 100, False, True, True)
datasource = gen_data_cam.get_data()

# initialize plot tool
fig = plt.figure(1)

error = np.zeros([len(datasource.images),3])

iterations = len(datasource.images)
iterations = 340
for i in range(iterations):
    np_image = datasource.images[i]
    feed={tf.get_default_graph().get_tensor_by_name('Placeholder:0'): np.expand_dims(np_image, axis=0) }

    # ground truth x y z
    pose_x= np.asarray(datasource.poses[i][0:2])

    # ground truth euler angles
    pose_q= np.asarray(datasource.poses[i][3:5]) 
    # pose_euler_angle = transforms3d.euler.quat2euler(pose_q)
    
    x_q = trainer.sess.run([tf.get_default_graph().get_tensor_by_name('fc9/fc9:0') ], feed)
    #  pdb.set_trace()

    # x y z
    pred_x = np.squeeze(x_q)[0:2]

    # euler angle
    pred_q = np.squeeze(x_q)[3:5]
    # pred_euler_angle = transforms3d.euler.quat2euler(pred_q)

    # scatter plot for pose
    plt.scatter(pose_x[1],pose_x[0],c='g')
    plt.scatter(pred_x[1],pred_x[0],c='r')
    plt.plot([pose_x[1],pred_x[1]],[pose_x[0],pred_x[0]],c='k')
    # plt.pause(0.01)
    plt.draw()
    error[i,:] = np.array([pose_x[1]-pred_x[1],pose_x[0]-pred_x[0],pose_q[-1]-pred_q[-1]])
    
    print("iteration {}\n".format(i))
    
# save the plot
plt.legend(['ground truth','prediction'])
fig.savefig(figname)

# calculate stddev and mean error
meanErr = np.sum(error,axis=0)/len(error)
stdErr = np.std(error,axis=0)
print("The mean error is {} and standard deviation is {}.".format(meanErr,stdErr))

goodId = [106,204]
badId = [298,321]
print(error[goodId[0],:],error[goodId[1],:])
print(error[badId[0],:],error[badId[1],:])
