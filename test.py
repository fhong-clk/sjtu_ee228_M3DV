import argparse
import json
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data

from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import metrics
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms

import pandas as pd

from livelossplot import PlotLosses


data_path = './'
model_path='model0.72316'
testpath='test/'
INPUT_DIM = 84
MAX_PIXEL_VAL = 255
'''
MEAN = 0
STDDEV = 0
MEAN_axial = 76.5
STDDEV_axial = 60.4
MEAN_coronal = 71.1
STDDEV_coronal = 65.4
MEAN_sagittal = 65.5
STDDEV_sagittal = 49.1
'''
mean=110.211324000001
std=78.31483481542116
class Dataset(data.Dataset):
    #diagnosis = 1 abnorm
    #diagnosis = 2 meniscus
    #diagnosis = 3 acl
    def __init__(self, use_gpu, train,test):
        super().__init__()
        self.use_gpu = use_gpu
        #self.part = part
        self.train=train
        label_dict = {}
        self.paths = []
        self.test=test
        
        data_path = '.'
        if train:
            path = os.path.join(data_path, 'train_val_pseudolabeling.csv')
            data_path = os.path.join(data_path, 'train_val')
        elif not test:
            path = os.path.join(data_path, 'val.csv')
            data_path = os.path.join(data_path, 'val')
        else:
            path = os.path.join(data_path, 'sampleSubmission.csv')
            data_path = testpath
        
        df = pd.read_csv(path, header=0)

        for item in df['name']:
            self.paths.append(os.path.join(data_path, str(item).zfill(4)) + '.npz')

        self.labels = np.array(df.iloc[:,1])

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]
        vol = np.load(path)
        vox=vol['voxel']
        seg=vol['seg']
        
        vol=vox
        '''
        a=np.where(seg!=0)
        b=np.array(a,dtype=int)
        segmin=b.min()
        segmax=b.max()
        print(segmax-segmin)
        '''
        #vol=vol['voxel']
        # crop middle
        #pad = int((vol.shape[2] - INPUT_DIM)/2)
        a=0
        #if self.train:
        #    a = random.randint(0,10)
        #vol = vol[segmin:segmax,segmin:segmax,segmin:segmax]
        #transform.resize(vol,(segmax-segmin,100,100))
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
        
        '''
        if self.part == "axial":
            MEAN = MEAN_axial
            STDDEV = STDDEV_axial
        elif self.part == "coronal":
            MEAN = MEAN_coronal
            STDDEV = STDDEV_coronal
        elif self.part == "sagittal":
            MEAN = MEAN_sagittal
            STDDEV = STDDEV_sagittal
        '''
        MEAN=mean
        STDDEV=std
        # normalize
        vol = (vol - MEAN) / STDDEV
        s=0
        #if self.train:
        #    s=np.random.normal(0, 0.1, vol.shape)
        vol=vol+s
        vol=vol*seg
      
        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)


        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])
        

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

  
use_gpu=True
valid_dataset = Dataset(use_gpu, False,True)
model = MRNet().cuda()
model.eval()
#model_path='model0.72316'
state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
model.load_state_dict(state_dict)
result=np.zeros([len(valid_dataset)])
for i in range(len(valid_dataset)):
    img_o = valid_dataset[i]
    #print(img_o[1])
    pred=float(torch.sigmoid(model.forward(img_o[0].cuda())))
    result[i]=pred
    #print(float(torch.sigmoid(model.forward(img_o[0].cuda()))))
testdata = pd.read_csv("sampleSubmission.csv")
testdata=np.asarray(testdata)
testid=[]
for i in range(testdata.shape[0]):
    testid.append(testdata[i][0])
relist=result.tolist()
#name=['pred']
resultcsv=pd.DataFrame({'name':testid,'predicted':relist})
#print(resultcsv)
resultcsv.to_csv('test.csv',index=False)