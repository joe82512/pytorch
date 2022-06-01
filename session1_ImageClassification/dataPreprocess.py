# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import torch
from torchvision import datasets, transforms

class FileOperation():
    def __init__(self):
        pass
    
    def shuffleData(self,x):
        x = np.array(x)
        np.random.seed(10)
        random_idx = np.arange(x.shape[0])
        np.random.shuffle(random_idx)
        return x[random_idx]

    def splitData(self,x,rate=0.1):
        x_train = x[int(x.shape[0]*rate):]
        x_val = x[:int(x.shape[0]*rate)]
        return x_train, x_val

    def createDatabase(self,data, data_type):
        # Ex: c1_path, train
        filename = data[0].split('\\') #windows path
        class_name = filename[-2]
        data_path = os.path.join(os.getcwd(),'database',str(data_type),str(class_name))
        
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
            
        for d in data:
            filename = d.split('\\') #windows path
            shutil.copy(d, os.path.join(data_path,filename[-1]))
    
    def getDatabase(self,path_list):
        for p in path_list:    
            p = self.shuffleData(p)
            train, val = self.splitData(p)
            self.createDatabase(train,'train')
            self.createDatabase(val,'val')





class TransformPytorchDataType():
    def __init__(self,batch_size,img_size=224,normalize=.5):
        self.batch_size = batch_size
        self.img_size = img_size
        self.normalize = normalize
    
    def trainDataLoader(self):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size), #對圖片尺寸做一個縮放切割
            transforms.RandomHorizontalFlip(), #水平翻轉
            transforms.ToTensor(),
            transforms.Normalize(
                (self.normalize,self.normalize,self.normalize),
                (self.normalize,self.normalize,self.normalize)
            ) #進行歸一化
        ])
        train_path = os.path.join(os.getcwd(),'database','train')
        train_datasets = datasets.ImageFolder(train_path, transform=train_transforms)
        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True)
        return train_dataloader
    
    def valDataLoader(self):
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (self.normalize,self.normalize,self.normalize),
                (self.normalize,self.normalize,self.normalize)
            )
        ])
        val_path = os.path.join(os.getcwd(),'database','val')
        val_datasets = datasets.ImageFolder(val_path, transform=val_transforms)
        val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=self.batch_size, shuffle=True)
        return val_dataloader
    