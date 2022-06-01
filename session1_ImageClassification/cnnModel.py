# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision import transforms
from torchsummary import summary #need pip install
from tqdm import tqdm #need pip install

class VGGNet(nn.Module):
    def __init__(self,num_classes=3,freeze=True): #num_classes
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True) #從預訓練模型加載VGG16網絡參數
        if freeze: #freeze parameter
            for param in net.parameters():
                param.requires_grad = False
        
        self.features = net.features #保留VGG16的特徵層
        '''
        # https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html
        self.classifier = nn.Sequential( #定義自己的分類層
                nn.Linear(512*7*7, 4096), #512*7*7不能改變，由VGG16網絡決定
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
        )
        '''
        # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        net.classifier[6] = nn.Linear(4096,num_classes)
        self.classifier = net.classifier
               

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x





class ModelOperation():
    def __init__(self,model):        
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
    
    def structure(self,shape):
        summary(self.model, shape)
        
    def save_model(self,filename):
        self.filename = filename
        
    def compiler(self,optimizer,loss_func):
        self.optimizer = optimizer
        self.loss_func = loss_func
        
    def fit(self,train_data,val_data,epoch):
        val_loss_min = np.Inf
        for i in range(epoch):
            train_acc,train_loss,val_acc,val_loss = 0.,0.,0.,0.
            print('epoch {}'.format(i + 1))
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # train
            self.model.train()    
            for img, label in tqdm(train_data):                
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(img) # forward propagation
                _, preds = torch.max(output, 1) # acc
                loss = self.loss_func(output, label) # loss
                # just for train
                loss.backward() # back propagation
                self.optimizer.step() # parameter update
                
                train_acc += torch.sum(preds == label.data)
                train_loss += loss.item()*img.size(0)        
            
            # val
            self.model.eval()
            for img, label in tqdm(val_data):
                img, label = img.to(self.device), label.to(self.device)
                output = self.model(img)
                loss = self.loss_func(output, label)
                _, preds = torch.max(output, 1)
                val_acc += torch.sum(preds == label.data)
                val_loss += loss.item()*img.size(0)
                
            
            train_acc = train_acc/len(train_data.dataset)
            train_loss = train_loss/len(train_data.dataset)
            val_acc = val_acc/len(val_data.dataset)
            val_loss = val_loss/len(val_data.dataset)
            
            print('\tTraining Acc: {:.2f}% \tTraining Loss: {:.2f}%'.format(train_acc*100, train_loss*100))
            print('\tValidation Acc: {:.2f}% \tValidation Loss: {:.2f}%'.format(val_acc*100, val_loss*100))
            
            # save model
            if val_loss <= val_loss_min:
                print('Saving model ... Validation loss decreased {:.2f}% --> {:.2f}%'.format(
                val_loss_min*100,
                val_loss*100))
                torch.save(self.model.state_dict(), self.filename)
                val_loss_min = val_loss
        
    def load_model(self):
        self.model.load_state_dict(torch.load(self.filename))
        
    def predict(self,test_data,img_size=224,normalize=.5):
        # https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch/notebook
        self.model.eval()
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (normalize,normalize,normalize),
                (normalize,normalize,normalize)
            )
        ])
        test_batch = torch.stack([val_transforms(img).to(self.device) for img in test_data])
        pred_logits_tensor = self.model(test_batch)
        self.pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().detach().numpy()
        return self.pred_probs
    
    def classifier(self,class_map):
        c_idx = np.argmax(self.pred_probs,axis=1)
        c_name = [class_map[i] for i in c_idx]
        return c_idx, c_name