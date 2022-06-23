# -*- coding: utf-8 -*-

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

#from .utils import layer_finders


class ResnetClassifier(nn.Module):
    def __init__(self, pretrain_resnet):
        super(ResnetClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.bn1 = pretrain_resnet.bn1
        self.relu = pretrain_resnet.relu
        self.maxpool = pretrain_resnet.maxpool
        self.layer1 = pretrain_resnet.layer1
        self.layer2 = pretrain_resnet.layer2
        self.layer3 = pretrain_resnet.layer3
        self.layer4 = pretrain_resnet.layer4

        self.avgpool = pretrain_resnet.avgpool  #AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear_final = nn.Linear(512, 2)
        #self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #  No adaptative average pooling
        #sprint(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape) #128
        x = self.maxpool(x)
        #print(x.shape) #64
        x = self.layer1(x)
        #print(x.shape) #64x64
        x = self.layer2(x)
        #print(x.shape) #32x32
        x = self.layer3(x)
        #print(x.shape) #16x16
        x = self.layer4(x)
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.shape[0], 512)
        #print(x.shape)
        x = self.linear_final(x)
        #print()
        #print(x.shape)

        #x = self.softmax(x)
        return x

class ResnetClassifierOriginal(nn.Module):
    def __init__(self, pretrain_resnet):
        super(ResnetClassifierOriginal, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = pretrain_resnet.bn1
        self.relu = pretrain_resnet.relu
        self.maxpool = pretrain_resnet.maxpool
        self.layer1 = pretrain_resnet.layer1
        self.layer2 = pretrain_resnet.layer2
        self.layer3 = pretrain_resnet.layer3
        self.layer4 = pretrain_resnet.layer4

        self.avgpool = pretrain_resnet.avgpool  #AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear_final = nn.Linear(512, 2)
        #self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #  No adaptative average pooling
        #sprint(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape) #128
        x = self.maxpool(x)
        #print(x.shape) #64
        x = self.layer1(x)
        #print(x.shape) #64x64
        x = self.layer2(x)
        #print(x.shape) #32x32
        x = self.layer3(x)
        #print(x.shape) #16x16
        x = self.layer4(x)
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.shape[0], 512)
        #print(x.shape)
        x = self.linear_final(x)
        #print()
        #print(x.shape)

        #x = self.softmax(x)
        return x


class ResnetClassifierOriginal3(nn.Module):
    def __init__(self, pretrain_resnet):
        super(ResnetClassifierOriginal3, self).__init__()
        self.conv1 = pretrain_resnet.conv1
        self.bn1 = pretrain_resnet.bn1
        self.relu = pretrain_resnet.relu
        self.maxpool = pretrain_resnet.maxpool
        self.layer1 = pretrain_resnet.layer1
        self.layer2 = pretrain_resnet.layer2
        self.layer3 = pretrain_resnet.layer3
        self.layer4 = pretrain_resnet.layer4

        self.avgpool = pretrain_resnet.avgpool  #AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear_final = nn.Linear(512, 2)
        #self.softmax = torch.nn.Softmax()

    def forward(self, x):
        #  No adaptative average pooling
        #sprint(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape) #128
        x = self.maxpool(x)
        #print(x.shape) #64
        x = self.layer1(x)
        #print(x.shape) #64x64
        x = self.layer2(x)
        #print(x.shape) #32x32
        x = self.layer3(x)
        #print(x.shape) #16x16
        x = self.layer4(x)
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.shape[0], 512)
        #print(x.shape)
        x = self.linear_final(x)
        #print()
        #print(x.shape)

        #x = self.softmax(x)
        return x




#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

