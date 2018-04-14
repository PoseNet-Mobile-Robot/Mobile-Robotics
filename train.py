import sys, os
#sys.path.insert(0, '/home/eecs568/miniconda3/envs/tensorflow/lib/python3.5/site-packages')
import data_handler
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import cv2, imutils
import vgg
import gen_data_cam as gen_data
import pdb

def delete_network_backups(filename_prefix):
    try:
        os.remove(filename_prefix+str(".index"))
    except OSError:
        print(OSError)
        pass
    try:
        os.remove(filename_prefix+str(".meta"))
    except OSError:
        pass
    try:
        os.remove(filename_prefix+str(".data-00000-of-00001"))
    except OSError:
        pass


class trainer():
    
    def __init__(self,path_to_weight, path_to_data, beta, use_quaternion=True, resume_training=False, just_test=False):
        self.network_input_size = 224
        self.output_dim = 7 if use_quaternion else 6
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if resume_training:
            self.restore_network(path_to_weight)
        else:
            self.image_inputs = tf.placeholder(tf.float32, [None, self.network_input_size, self.network_input_size, 3])
            self.label_inputs = tf.placeholder(tf.float32, [None, self.output_dim])  # [ X Y Z W  P Q R]

            self.network = vgg.VGG16({'data': self.image_inputs})
            self.regen_regression_network()
            self.build_loss(beta)
            self.saver = tf.train.Saver()
            
        self.merged_summary = tf.summary.merge_all()
        now = datetime.now()
        self.summary_now = now.strftime("%Y%m%d-%H%M%S")
        #self.train_writer = tf.summary.FileWriter('./summary/train/', self.sess.graph)
        self.train_writer = tf.summary.FileWriter('./summary/train'+ self.summary_now  + "/", self.sess.graph)
        self.test_writer = tf.summary.FileWriter( './summary/test'+ now.strftime("%Y%m%d-%H%M%S") + "/") 

        # initialize
        if just_test == False:
            self.init_data_handler(path_to_data)        
        self.init_op = tf.global_variables_initializer() # tf.variables_initializer(self.init_vars )

        if not resume_training:
            self.sess.run(self.init_op)
            self.load_weight(path_to_weight)
            print("Model initialized")

    def init_data_handler(self,path_to_data):
        #self.data_handler = data_handler.Process(path_to_data, 'dataset_train.txt', False)
        self.data_handler = gen_data.get_data()

    def load_weight(self,path_to_weight):
        self.network.load(path_to_weight, self.sess)
        #self.saver.restore(self.sess, path_to_weight)
        print("Model Restored")

    def regen_regression_network(self):
        graph = tf.get_default_graph()
        fc7 = graph.get_tensor_by_name("fc7/fc7:0")

        with tf.variable_scope("fc8", reuse=tf.AUTO_REUSE) as scope:
            fc8_input_shape = fc7.get_shape()
            feed_in, dim = (fc7, fc8_input_shape[-1].value)
            fc8_weights = tf.get_variable('weights', [dim, 2048])
            self.network.variable_summaries( fc8_weights,  "_weights_fc" )
            fc8_biases =  tf.get_variable('biases', [2048])
            op = tf.nn.xw_plus_b
            fc8 = op(feed_in, fc8_weights, fc8_biases, name=scope.name)

        with tf.variable_scope("fc9", reuse=tf.AUTO_REUSE) as scope:
            fc9_input_shape = fc8.get_shape()

            feed_in, dim = (fc8, fc9_input_shape[-1].value)
            fc9_weights = tf.get_variable('weights', [dim, self.output_dim])
            self.network.variable_summaries( fc9_weights,  "_weights_fc" )
            fc9_biases =  tf.get_variable('biases', [self.output_dim])
            op = tf.nn.xw_plus_b
            fc9 = op(feed_in, fc9_weights, fc9_biases, name=scope.name)
        self.init_vars = [fc8_weights, fc8_biases , fc9_weights,  fc9_biases]
        self.regression_out = fc9
        
        tf.identity(self.regression_out, name="regression_output")
        self.network.variable_summaries(self.regression_out, "regression_output_")

    def restore_network(self, path_to_weight):
        self.saver = tf.train.import_meta_graph(path_to_weight + ".meta" )
        graph = tf.get_default_graph()
        self.regression_out = tf.get_default_graph().get_tensor_by_name('fc9/fc9:0')
        self.loss = graph.get_operation_by_name("final_loss")
        self.train_op = tf.get_default_graph().get_operation_by_name("Adam_minimizer")
        self.saver.restore(self.sess, path_to_weight)#tf.train.latest_checkpoint('./'))
        self.image_inputs = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
        self.label_inputs = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')

        print("Model restored.")
        
    def build_loss(self, beta=100):
        self.translation_loss = tf.sqrt(tf.nn.l2_loss(self.regression_out[0:3] - self.label_inputs[0:3]))
        self.rotation_loss = tf.sqrt(tf.nn.l2_loss( self.regression_out[3:] - self.label_inputs[3:]  ))
        self.loss = self.translation_loss + beta * self.rotation_loss
        tf.identity(self.loss, name="final_loss")

        self.optimizer = tf.train.AdamOptimizer( learning_rate=0.00001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False)
        slot_var_names = self.optimizer.get_slot_names()
        for v in tf.trainable_variables():
            for i in slot_var_names:
                self.init_vars.append(self.optimizer.get_slot(v, i))
        train_vars = tf.trainable_variables()
        del train_vars[-6]
        del train_vars[-5]
        self.compute_gradients = self.optimizer.compute_gradients (self.loss, train_vars)
        self.train_op = self.optimizer.apply_gradients(self.compute_gradients , name='Adam_minimizer')
        #self.train_op = self.optimizer.minimize(self.loss,  name='Adam_minimizer')
        grad_summ_op = tf.summary.merge([tf.summary.histogram("%s_grad" % g[1].name, g[0]) for g in self.compute_gradients ])
        self.network.variable_summaries(self.translation_loss, "translation_loss_")
        self.network.variable_summaries(self.rotation_loss, "rotation_loss_")
        self.network.variable_summaries(self.loss, "final_weighted_loss_")

    def test(self, img, need_rotate_angle=270, num_random_crops=20):
        pdb.set_trace()
        if img.shape[2] != 3:
            print ("We only accept 3-dimensional rgb images")
        if img.shape[0] > img.shape[1]:
            img = imutils.rotate(img, need_rotate_angle)
            img = imutils.resize(img , height=256)

        input_size = self.network_input_size # 224 here
        input_batch = np.zeros((num_random_crops,input_size,input_size,3))
        if num_random_crops == 1:
            rand_range = [img.shape[0]-input_size, img.shape[1]-input_size] # height, width
            for index in range(num_random_crops):
                h = np.random.randint(rand_range[0])
                w = np.random.randint(rand_range[1])
                input_batch[index, :] = img[h:h+input_size, w:w+input_size, :]
                t_r_output = self.sess.run([self.regression_out],
                                   feed_dict={self.image_inputs: input_batch})
            return np.mean(t_r_output, axis=0)
        else:
            tf_output = self.sess.run([self.regression_out],
                                      feed_dict={self.image_inputs: gen_data.centeredCrop(img, input_size)} )
            return tf_output

    
    
    def train(self, batch_size, epochs):
        
        total_loss = 0
        total_batch = 281 #int(self.data_handler.numimages() * self.data_handler.genNum * 1.0 / batch_size) #100
        if total_batch==0:
            pdb.set_trace()
        #print("[trainer] Start Training, size of dataset is " +str(self.data_handler.numimages() * self.data_handler.num_crops ))
        #pdb.set_trace()
        for epoch in range(epochs):
            #self.data_handler.reset()
            #self.data_handler.generateData(500)
            data_gen = gen_data.gen_data_batch(self.data_handler )
            for i in range(total_batch):
                
                #data_runout_flag, one_batch_image , one_batch_label = self.data_handler.fetch(batch_size)
                '''
                if data_runout_flag == False:
                    if self.data_handler.remimages() > 0:
                        self.data_handler.generateData(500)
                    else:
                        self.data_handler.reset()
                        self.data_handler.generateData(500)
                    data_runout_flag, one_batch_image , one_batch_label = self.data_handler.fetch(batch_size)
                '''
                one_batch_image, np_poses_x, np_poses_q = next(data_gen)
                one_batch_label = np.hstack((np_poses_x, np_poses_q))
                feeds ={self.image_inputs: one_batch_image, self.label_inputs: one_batch_label }
                summary, loss, gradients = self.sess.run([self.merged_summary, self.loss, self.compute_gradients ], feed_dict=feeds) 
                self.sess.run([self.train_op], feed_dict=feeds )
                print("[Epoch "+str(epoch)+" trainer] Train one batch of size "+str(batch_size)+", loss is "+str(loss))
                total_loss += loss
                self.train_writer.add_summary(summary, epoch * total_batch + i)

            avg_loss = (total_loss)/total_batch
            self.saver.save(self.sess, "./"+self.summary_now+"model_epoch_"+str(epoch)+".ckpt")
            if epoch > 0: delete_network_backups("./"+self.summary_now+"model_epoch_"+str(epoch-1)+".ckpt" )
            print("[trainer] Epoch " + str(epoch )+ " ends, avg loss =" + "{:.3f}".format(avg_loss))

            total_loss = 0
        

if __name__ == "__main__":
    argv = sys.argv
    if len(sys.argv) < 5:
        argv = ['' for _ in range(6)]
        argv[1] = './vgg.data'
        argv[2] = './ShopFacade/'
        argv[3] = 100
        argv[4] = True
        argv[5] = bool(int(False))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_thread1 = trainer(argv[1], argv[2], 100, use_quaternion=argv[4], resume_training=False )
    train_thread1.train(32, 10)
