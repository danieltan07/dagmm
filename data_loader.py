import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        data = pd.read_csv(data_path, header=None,names=['duration', 
            'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'type'])

        self.mode=mode

        data.loc[data["service"].str.startswith('http', na=False),"service"] = "http"
        data.loc[data["type"] != "normal.", 'type'] = 1
        data.loc[data["type"] == "normal.", 'type'] = 0

        one_hot_protocol = pd.get_dummies(data["protocol_type"])
        one_hot_service = pd.get_dummies(data["service"])
        one_hot_flag = pd.get_dummies(data["flag"])

        data = data.drop("protocol_type",axis=1)
        data = data.drop("service",axis=1)
        data = data.drop("flag",axis=1)
        



        data = pd.concat([one_hot_protocol, one_hot_service,one_hot_flag, data],axis=1)

        self.normal_data = data[data["type"] == 0]

        cols_to_norm = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", 
            "hot", "num_failed_logins", "num_compromised", "root_shell", "su_attempted", "num_root", 
            "num_file_creations", "num_shells", "num_access_files", "count", "srv_count", 
            "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate" ]

        # self.normal_data[cols_to_norm] = self.normal_data[cols_to_norm].apply(lambda x: (x - x.mean()) / (1 if x.std() == 0 else x.std()))

        self.attack_data = data[data["type"] == 1]
        self.normal_data = self.normal_data.drop("type", axis=1)
        self.attack_data = self.attack_data.drop("type", axis=1)

        N, D = self.normal_data.shape

        N_a, _ = self.attack_data.shape

        randIdx = np.arange(N_a)
        np.random.shuffle(randIdx)

        self.attack_data = self.attack_data.iloc[randIdx[:N//4]]
        
        randIdx = np.arange(N)
        np.random.shuffle(randIdx)
        tr_max = int((N*1.25)//2)
        trainIdx = randIdx[:tr_max]
        testIdx = randIdx[tr_max:]

        self.train_normal = self.normal_data.iloc[trainIdx]
        self.test_normal = self.normal_data.iloc[testIdx]

        self.train_means = self.train_normal[cols_to_norm].mean()
        self.train_std = self.train_normal[cols_to_norm].std()
        self.train_min = self.train_normal[cols_to_norm].min()
        self.train_max = self.train_normal[cols_to_norm].max()

        # self.train_normal.loc[:,cols_to_norm] = (self.train_normal[cols_to_norm] - self.train_means[cols_to_norm]) /  (self.train_std[cols_to_norm] + 1e-8)
        # self.test_normal.loc[:,cols_to_norm] = (self.test_normal[cols_to_norm] - self.train_means[cols_to_norm]) /  (self.train_std[cols_to_norm] + 1e-8)
        # self.attack_data.loc[:,cols_to_norm] = (self.attack_data[cols_to_norm] - self.train_means[cols_to_norm]) /  (self.train_std[cols_to_norm] + 1e-8)


        self.train_normal.loc[:,cols_to_norm] = (self.train_normal[cols_to_norm] - self.train_min[cols_to_norm]) /  (self.train_max[cols_to_norm] - self.train_min[cols_to_norm] + 1e-8)
        self.test_normal.loc[:,cols_to_norm] = (self.test_normal[cols_to_norm] - self.train_min[cols_to_norm]) /  (self.train_max[cols_to_norm] - self.train_min[cols_to_norm] + 1e-8)
        self.attack_data.loc[:,cols_to_norm] = (self.attack_data[cols_to_norm] - self.train_min[cols_to_norm]) /  (self.train_max[cols_to_norm] - self.train_min[cols_to_norm] + 1e-8)


        self.train_normal = self.train_normal.as_matrix()
        self.test_normal = self.test_normal.as_matrix()
        self.attack_data = self.attack_data.as_matrix()


        self.combined = np.concatenate([self.train_normal,self.test_normal,self.attack_data])
        self.combined_labels = np.zeros((self.combined.shape[0],))
        self.combined_labels[N:] = 1
        # print(self.combined_labels.shape)
        # print(N)
        # print(self.attack_data.shape)
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train_normal.shape[0]
        else:
            return self.combined.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train_normal[index])
        else:
           return np.float32(self.combined[index]), np.float32(self.combined_labels[index])
        

def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    dataset = KDD99Loader(data_path, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
