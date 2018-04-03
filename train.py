import sys, os
sys.path.insert(0, '/home/eecs568/miniconda3/envs/tensorflow/lib/python3.5/site-packages')
import preprocess
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import cv2, imutils
import vgg
#import data
import pdb

def delete_network_backups(filename_prefix):
    try:
        os.remove(filename_prefix+str(".index"))
    except OSError:
        print OSError
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
    
    def __init__(self,path_to_weight, path_to_data, resume_training=False):
        self.network_input_size = 224
        self.image_inputs = tf.placeholder(tf.float32, [None, self.network_input_size, self.network_input_size, 3])
        self.label_inputs = tf.placeholder(tf.float32, [None, 7])  # [ X Y Z W  P Q R]
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if resume_training:
            self.restore_network(path_to_weight)
        else:
            self.network = vgg.VGG16({'data': self.image_inputs})
            self.regen_regression_network()
            self.build_loss()
            self.saver = tf.train.Saver()
            
        self.merged_summary = tf.summary.merge_all()
        now = datetime.now()
        self.train_writer = tf.summary.FileWriter('./summary/train'+ now.strftime("%Y%m%d-%H%M%S") + "/", self.sess.graph)
        self.test_writer = tf.summary.FileWriter( './summary/test'+ now.strftime("%Y%m%d-%H%M%S") + "/") 

        # initialize 
        self.init_data_handler(path_to_data)        
        self.init_op = tf.global_variables_initializer() # tf.variables_initializer(self.init_vars )

        if not resume_training:
            self.sess.run(self.init_op)
            self.load_weight(path_to_weight)
            print("Model initialized")

    def init_data_handler(self,path_to_data):
        self.data_handler = preprocess.preprocess(path_to_data)

    def load_weight(self,path_to_weight):
        self.network.load(path_to_weight, self.sess)
        #self.saver.restore(self.sess, path_to_weight)
        print("Model Restored")

    def regen_regression_network(self):
        graph = tf.get_default_graph()
        fc_out = graph.get_tensor_by_name("fc7/fc7:0")

        with tf.variable_scope("fc8", reuse=tf.AUTO_REUSE) as scope:
            input_shape = fc_out.get_shape()

            feed_in, dim = (fc_out, input_shape[-1].value)
            weights = tf.get_variable('weights', [dim, 7])
            self.network.variable_summaries( weights,  "_weights_fc" )
            biases =  tf.get_variable('biases', [7])
            self.init_vars = [weights, biases]

            op = tf.nn.xw_plus_b
            fc8 = op(feed_in, weights, biases, name=scope.name)
        self.regression_out = fc8
        tf.identity(self.regression_out, name="regression_output")
        self.network.variable_summaries(self.regression_out, "regression_output_")

    def restore_network(self, path_to_weight):
        
        self.saver = tf.train.import_meta_graph(path_to_weight + ".meta" )
        graph = tf.get_default_graph()
        self.regression_out = graph.get_operation_by_name("regression_output")
        self.loss = graph.get_operation_by_name("final_loss")
        self.train_op = tf.get_default_graph().get_operation_by_name("Adam_minimizer")
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        print("Model restored.")
        
    def build_loss(self, beta=300):
        self.translation_loss = tf.nn.l2_loss(self.regression_out[0:2] - self.label_inputs[0:2])
        self.rotation_loss = tf.nn.l2_loss( self.regression_out[3:6]/tf.norm(self.regression_out[3:6]) - self.label_inputs[3:6]  )
        self.loss = self.translation_loss + beta * self.rotation_loss
        tf.identity(self.loss, name="final_loss")

        self.optimizer = tf.train.AdamOptimizer( learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False)
        slot_var_names = self.optimizer.get_slot_names()
        for v in tf.trainable_variables():
            for i in slot_var_names:
                self.init_vars.append(self.optimizer.get_slot(v, i))
        self.compute_gradients = self.optimizer.compute_gradients (self.loss ) #, tf.trainable_variables())
        self.train_op = self.optimizer.apply_gradients(self.compute_gradients , name='Adam_apply_gradients')
        #self.train_op = self.optimizer.minimize(self.loss,  name='Adam_minimizer')
        self.network.variable_summaries(self.translation_loss, "translation_loss_")
        self.network.variable_summaries(self.rotation_loss, "rotation_loss_")
        self.network.variable_summaries(self.loss, "final_weighted_loss_")

    # TODO: for each layer's weight && bias, add summaries
    def plot_gradients_each_layer(self, gradients, train_summary):
        for g, v in gradients:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                train_summary.append(grad_hist_summary)
                train_summary.append(sparsity_summary)
                tf.summary.merge(train_summary)

    
    def test(self, img, num_random_crops=10):
        if img.shape[2] != 3:
            print ("We only accept 3-dimensional rgb images")
        if img.shape[0] != self.network_input_size or img.shape[1] != self.network_input_size:
            if img.shape[0] < img.shape[1]:
                img = imutils.resize(img , height=256, interpolation=cv2.INTER_CUBIC)
            else:
                img = imutils.resize(img, width=256, interpolation=cv2.INTER_CUBIC)
        input_size = self.network_input_size # 224 here
        input_batch = np.zeros((num_random_crops,input_size,input_size,3))
        rand_range = [img.shape[0]-input_size, img.shape[1]-input_size] # height, width
        for index in range(num_random_crops):
            h = np.random.randint(rand_range[0])
            w = np.random.randint(rand_range[1])
            input_batch[index, :] = img[h:h+input_size, w:w+input_size, :]
        t_r_output = self.sess.run([self.regression_out],
                                   feed_dict={self.image_inputs: input_batch})
        return np.mean(t_r_output, axis=0)
    
    def train(self, batch_size, epochs):
        total_loss = 0
        total_batch = int(self.data_handler.numSamples() / batch_size)
        print("[trainer] Start Training, size of dataset is "+str(self.data_handler.numSamples()))
        #pdb.set_trace()
            
        for epoch in range(epochs):
            for i in range(total_batch):
                one_batch_image , one_batch_label = self.data_handler.fetch(batch_size)
                #summary, loss, gradients = self.sess.run([self.merged_summary, self.loss, self.compute_gradients ], 
                #                feed_dict={self.image_inputs: one_batch_image, self.label_inputs: one_batch_label })

                #self.sess.run([self.optimizer.apply_gradients], feed_dict={gradients})
                feeds ={self.image_inputs: one_batch_image, self.label_inputs: one_batch_label }
                pdb.set_trace()
                summary, loss,grad,  _ = self.sess.run([self.merged_summary, self.loss, self.compute_gradients, self.train_op], feeds)
                self.plot_gradients_each_layer( grad, summary )
                print("[Epoch "+str(epoch)+" trainer] Train one batch of size "+str(batch_size)+", loss is "+str(loss))
                total_loss += loss
                self.train_writer.add_summary(summary, epoch * total_batch + i)
            avg_loss = (total_loss)/total_batch
            self.saver.save(self.sess, "./model_epoch_"+str(epoch)+".ckpt")
            if epoch > 0: delete_network_backups("./model_epoch_"+str(epoch-1)+".ckpt" )
            print("[trainer] Epoch " + str(epoch )+ " ends, avg loss =" + "{:.3f}".format(avg_loss))
            self.data_handler.reset()
            total_loss = 0
        

if __name__ == "__main__":
    argv = sys.argv
    if len(sys.argv) < 4:
        argv = ['', '', '', '']
        argv[1] = './vgg.data'
        argv[2] = './ShopFacade/'
        argv[3] = bool(int(False))
    
    train_thread = trainer(argv[1], argv[2], bool(int(argv[3])))
    train_thread.train(50, 600)
