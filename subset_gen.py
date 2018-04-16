import os, sys, csv, shutil
from tqdm import tqdm
from numpy import genfromtxt
import numpy as np
import pdb

class Subset:
    def __init__(self, folder_name, label_file,  train_freq, test_freq, tail_str, name_timestep_len, accuracy=1e-5):
        self.imgs = []
        self.timestep_len = name_timestep_len
        self.folder_name = folder_name
        self.train_freq = train_freq
        self.test_freq = test_freq
        self.train_folder = folder_name + '/train' 
        self.test_folder = folder_name  + '/test'
        self.tail_str = tail_str
        self.name_has_prefix = None
        
        for name in sorted(os.listdir(folder_name)):
            if name[-len(tail_str):] != tail_str:
                continue
            if name[0:5] == 'frame':
                name = name[5:]
                self.name_has_prefix = 'frame'
            self.imgs.append(name)
        self.labels = genfromtxt(label_file,  delimiter=',')

        # pointer point to the current timestamp that wait to be matched
        self.currentMatch = 0
        # match tolerance in microseconds
        self.matchTol = 1.0/accuracy

                
    def gen_subset(self, is_euler= False, is_train=True, is_test=True):
        if is_train:
            self.dump_to_folder( self.train_freq, self.train_folder,'dataset_train.csv',  True, is_euler)
        if is_test:
            self.dump_to_folder( self.test_freq, self.test_folder,'dataset_test.csv',  False,is_euler)
        
    def dump_to_folder(self, freq, new_folder, new_label_file, is_train, is_euler):
        table_ind = 0
        total_num_imgs = len(self.imgs)
        table = np.zeros((total_num_imgs // freq , self.labels.shape[1]))
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_labels = open(new_folder + '/' + new_label_file, "w")
        for i in tqdm(range(total_num_imgs)):
            if (i % freq == 0):
                img_i = self.imgs[i][:-len(self.tail_str)] # only contain number 
                label_i = self.match(img_i)
                if label_i == -1: continue
                table[table_ind, :] = self.labels[label_i,:]
                if is_train:
                    table[table_ind, 0] = str(int(self.labels[label_i, 0]))[0:self.timestep_len]
                    img_i = img_i[0:self.timestep_len]

                name = self.name_has_prefix + self.imgs[i] if self.name_has_prefix is not None else self.imgs[i]
                    
                shutil.copyfile(self.folder_name +'/' + name , new_folder + '/' + img_i + self.tail_str)
                to_write = str(int(table[table_ind, 0]))+ ','
                for i in range(1,table.shape[1]-1):
                    to_write = to_write +  str(float(table[table_ind, i])) + ','
                to_write  = to_write + str(float(table[table_ind, table.shape[1]-1]))+ '\n' 
                new_labels.write(to_write)
                table_ind += 1
                if table_ind == table.shape[0]: break
        new_labels.close()
        #np.savetxt( , table, delimiter=",")


    def match(self,str_timestamp):
        timestamp = int(str_timestamp)

        matchId = -1
        # begin match
        for i in range(len(self.labels)):
            if(abs(timestamp-self.labels[self.currentMatch,0])>self.matchTol):
                if(self.currentMatch>=len(self.labels)-1):
                    self.currentMatch = 0
                else:
                    self.currentMatch += 1
            else:
                matchId = self.currentMatch
                if(self.currentMatch>=len(self.labels)-1):
                    self.currentMatch = 0
                else:
                    self.currentMatch += 1
                
        return matchId
