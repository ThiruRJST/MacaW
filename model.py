import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import numpy as np
import timm
import h5py
from efficientnet_pytorch import EfficientNet



class custom_effnet(nn.Module):
    def __init__(self):
        super(custom_effnet,self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.model._conv_stem.in_channels = 1
        weight = self.model._conv_stem.weight.mean(1,keepdim=True)
        self.model._conv_stem.weight = torch.nn.Parameter(weight)
        print(self.model)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs,24)

    def forward(self,x):
        return self.model(x)

class custom_resnet(nn.Module):
    def __init__(self):
        super(custom_resnet,self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,24)

    def forward(self,x):
        return self.model(x)

class ConvMod(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,act=True,strides=1):
        super(ConvMod,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=strides)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act
    
    def forward(self,x):
        if self.act is True:
            x = F.relu(self.bn(self.conv(x)))
        else:
            x = self.bn(self.conv(x))

        return x

class CNN_14(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(CNN_14,self).__init__()
        self.conv1 = nn.Sequential(
            ConvMod(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size),
            ConvMod(64,64,kernel_size=kernel_size)
        )
        
        self.conv2 = nn.Sequential(
            ConvMod(64,128,kernel_size=kernel_size),
            ConvMod(128,128,kernel_size=kernel_size)
        )
        self.conv3 = nn.Sequential(
            ConvMod(128,256,kernel_size=kernel_size),
            ConvMod(256,256,kernel_size=kernel_size)

        )
        self.conv4 = nn.Sequential(
            ConvMod(256,512,kernel_size=kernel_size),
            ConvMod(512,512,kernel_size=kernel_size)
        )
        self.pool = nn.AvgPool2d(kernel_size=(2,2))
        self.linear = nn.Linear(512,512)
        self.classifier = nn.Linear(512,24)
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = F.adaptive_avg_pool2d(x,(1,1))
        x = x.view(-1,512)
        x = self.classifier(x)
        return x


