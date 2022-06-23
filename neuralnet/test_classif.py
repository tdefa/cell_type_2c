#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:22:18 2021

@author: thomas
"""

import argparse
import warnings
from dataset_classif_point import SpotDataset
from resnet_extractor import ResnetClassifier, ResnetClassifierOriginal, LeNet5, ResnetClassifierOriginal3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from pathlib import Path


def test_model(model, criterion,  dataloaders, device):
    since = time.time()
    model = model.eval()
    list_pred = []
    list_labels = []

    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels["label"].to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        list_pred += list(preds.numpy())
        list_labels += list(labels.data.numpy())

    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects / len(dataloaders.dataset)
    print(len(dataloaders.dataset))
    #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
      #  "val", epoch_loss, epoch_acc))
    tp = sum(np.array(list_pred) * np.array(list_labels))
    p = sum(list_labels)
    tn = sum(~np.array(list_pred).astype(bool) * ~np.array(list_labels).astype(bool))
    n = sum(~np.array(list_labels).astype(bool))
    diff = np.array(list_labels) - np.array(list_pred)
    fp = len(np.where(diff == -1)[0])
    fn = len(np.where(diff == 1)[0])
    return list_pred, list_labels, [p, n, tp, tn, fp, fn]


class MySampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    #data_source: Sized

    def __init__(self, test_indice):
        self.test_indice = test_indice

    def __iter__(self):
        return iter(test_indice)

    def __len__(self) -> int:
        return len(test_indice)

def get_indice(dataset):
    indice_zero = []
    indice_one = []

    for i in range(len(dataset)):
        if dataset[i][1]['label'] == 0:
            indice_zero.append(i)
        else:
            indice_one.append(i)
    return [indice_zero, indice_one]

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ptd', '--path_to_data',  type=str, default= "/home/thomas/Bureau/phd/label_dataset", metavar='N', help='')

    parser.add_argument('-bz', '--batch_size',  type=int, default= 8, metavar='N', help='')
    parser.add_argument('--nb_workers',  type=int, default = 4, metavar='N', help='')
    parser.add_argument('--balanced',  type=int, default = 1, metavar='N', help='')
    parser.add_argument('--sample_iter',  type=int, default = 5, metavar='N', help='')




    args = parser.parse_args()


    name_list =  [
            "/mki67/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy5",
            "/mki67/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy5",]
         
                          





    path_to_model = ["/home/thomas/Bureau/phd/first_lustra/model/new2603/0fra-Cy3_06modelbestlampserpine10",
                     "/home/thomas/Bureau/phd/first_lustra/model/new2603/0fra-Cy3_06modelbestlampserpine10",
                     "/home/thomas/Bureau/phd/first_lustra/model/new2603/0fra-Cy3_06modelbestlampserpine10",
                     "/home/thomas/Bureau/phd/first_lustra/model/new2603/0fra-Cy3_06modelbestlampserpine10",
                     "/home/thomas/Bureau/phd/first_lustra/model/new2603/0fra-Cy3_06modelbestlampserpine10",]

    for i in range(len(name_list)):
                   name_list[i] = args.path_to_data + name_list[i]




    p = Path('/home/thomas/Bureau/phd/first_lustra/model/new2603/lenetcy3norm8')
    print(p)
    #s = 'last'
    #path_to_model = [str(pp) for pp in list(p.glob(f'*{s}*'))]
    #path_to_model.sort()
    print(path_to_model)


    data_transforms = {
        'train': transforms.Compose([
            transforms.Normalize([0.456, 0.406], [ 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Normalize([ 0.456, 0.406], [ 0.224, 0.225])
        ]),
        }
    dico_res = {}
    dico_dico_res = {}
    warnings.filterwarnings("ignore")

    for model_index in range(len(path_to_model)):
        args.path_to_data =name_list[model_index]
        print(args.path_to_data)
        print()
        dico_dico_res[args.path_to_data] = []

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(device)
        criterion = nn.CrossEntropyLoss()

        full_dataset = SpotDataset(args.path_to_data,
                                   data_transforms['val'],
                                   normalize = 1,
                                   resize_size = 64,
                                   add_channel = 0)

        if args.balanced:
            for sample_index in range(args.sample_iter):
                indice_zero, indice_one = get_indice(full_dataset)
                nb_sample = min([len(indice_zero), len(indice_one)])
                random.seed(sample_index+model_index)
                test_indice = random.sample(indice_zero, nb_sample) + random.sample(indice_one, nb_sample)
                #print(test_indice)
    
                sampler = MySampler(test_indice)
                if len(sampler)==0:
                    print()
                    print("no sample")
                    print()
                    continue

                dataloaders =  torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
                                                         shuffle=False, num_workers=args.nb_workers, sampler = sampler)
    
                #model = LeNet5().double()
                model = ResnetClassifierOriginal(torch.load('resnet18_pretrained.pt')).double()
                #model = ResnetClassifier(torch.load('resnet18_pretrained.pt')).double()
                print(path_to_model[model_index])

                model.load_state_dict(torch.load(path_to_model[model_index], map_location=torch.device('cpu')))
                preds, labels,[p, n, tp, tn, fp, fn]  = test_model(model, criterion, dataloaders = dataloaders, device = device)
                dico_res[args.path_to_data] = [preds, labels,[p, n, tp, tn, fp, fn]]
                dico_dico_res[args.path_to_data].append(dico_res[args.path_to_data])
                print("positive sample %s" % int(p))
                print("negative sample %s" % int(n))
                print("Acc %s " % str((tn+tp)/(p+n)))
                print("tpr %s " % str(tp/p))
                print("tnr %s" % str(round(tn/n,4)))
                print("precision %s" % str((round(tp/(tp+fp), 4))))
        else:

                dataloaders =  torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.nb_workers)
    
                #model = LeNet5().double()
                print(path_to_model[model_index])
                model = ResnetClassifierOriginal(torch.load('resnet18_pretrained.pt')).double()
                model.load_state_dict(torch.load(path_to_model[model_index], map_location=torch.device('cpu')))
                preds, labels,[p, n, tp, tn, fp, fn]  = test_model(model, criterion, dataloaders = dataloaders, device = device)
                dico_res[args.path_to_data] = [preds, labels,[p, n, tp, tn, fp, fn]]
                print("positive sample %s" % int(p))
                print("negative sample %s" % int(n))
                print("Acc %s " % str((tn+tp)/(p+n)))
                print("tpr %s " % str(tp/p))
                print("tnr %s" % str(round(tn/n,2)))
                print("precision %s" % str((round(tp/(tp+fp), 2))))



#%%

if  not args.balanced:
    array = np.zeros(6, dtype='int')
    for k in dico_res:
        print(dico_res[k][-1])
        array  += np.array(dico_res[k][-1])
        print(array)
    [p, n, tp, tn, fp, fn] = array
    print("positive sample %s" % str(int(p)))
    print("negative sample %s" % str(int(n)))
    
    print("Acc %s " % str(round((tn+tp)/(p+n), 4)))
    print("tpr %s " % str(round(tp/p, 4)))
    print("tnr %s" % str(round(tn/n,4)))
    print("precision %s" % str((round(tp/(tp+fp), 4))))

if args.balanced:
    dico_var = {}

    array_global = np.zeros(6, dtype='float')
    ind = 0
    for k in dico_dico_res:
        dico_var[k] = []
        print(k)
        if len(dico_dico_res[k]) == 0:
            continue
        array = np.zeros(6, dtype='int')
    
        for dic in dico_dico_res[k]:
            print(dic[-1])
            array  += np.array(dic[-1])
            print(array)
            [p, n, tp, tn, fp, fn] = dic[-1]
            [acc, tpr, tnr, precision] = [(tn+tp)/(p+n), tp/p, tn/n, tp/(tp+fp)]
            dico_var[k].append([acc, tpr, tnr, precision])

        [p, n, tp, tn, fp, fn] = (array/len(dico_dico_res[k])).astype(float) #avant c'etait as i
        array_global += np.array([p, n, tp, tn, fp, fn])
        [vacc, vtpr, vtnr, vprecision] = np.var(np.array(dico_var[k]), axis = 0)
        print([vacc, vtpr, vtnr, vprecision])
        print("positive sample %s" % str(int(p)))
        print("negative sample %s" % str(int(n)))
    
        print("Acc %s, %s " % (str(round((tn+tp)/(p+n), 2)), format(vacc,'.2e')))
        print("tpr %s %s" % (str(round(tp/p, 2)), format(vtpr,'.2e')))
        print("tnr %s %s" % (str(round(tn/n,2)), format(vtnr,'.2e')))
        print("precision %s" % str((round(tp/(tp+fp), 2), format(vprecision,'.2e'))))

        print("positive sample %s" % str(int(p)))
        print("negative sample %s" % str(int(n)))

        print("Acc %s " % str(round((tn+tp)/(p+n), 2)))
        print("tpr %s " % str(round(tp/p, 2)))
        print("tnr %s" % str(round(tn/n,2)))
        print("precision %s" % str((round(tp/(tp+fp), 2))))

        print()
    
    [p, n, tp, tn, fp, fn] = array_global
    print([p, n, tp, tn, fp, fn])

    print("positive sample %s" % str(int(p)))
    print("negative sample %s" % str(int(n)))
    
    print("Acc %s " % str(round((tn+tp)/(p+n), 4)))
    print("tpr %s " % str(round(tp/p, 4)))
    print("tnr %s" % str(round(tn/n,4)))
    print("precision %s" % str((round(tp/(tp+fp), 4))))
    


#%%
sampler = MySampler(test_indice)
dataloaders =  torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.nb_workers,sampler = sampler)

model = ResnetClassifierOriginal(torch.load('resnet18_pretrained.pt')).double()
model.load_state_dict(torch.load("/home/thomas/Bureau/phd/first_lustra/model/r18oricy3116410/8modellastr18oricy3116410", map_location=torch.device('cpu')))

preds, labels, [p, n, tp, tn, fp, fn] = test_model(model, criterion, dataloaders = dataloaders, device = device)
#%%
for inputs, labels in dataloaders:
    inputs = inputs.to(device)


    labels = labels["label"].to(device)
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(preds)):
            if labels[i] == 0 and preds[i] == 1:
                fig, ax = plt.subplots(1,2,  figsize=(10,5))
                ax[0].imshow(inputs[i][0][1:63,1:63])
                ax[0].set_title("cy3")
                fig.suptitle(" prediction " + str(preds[i]) + " ; real label " + str(labels[i]))
                ax[1].set_title("cy5")

                ax[1].imshow(inputs[i][1][1:63,1:63])
                plt.show()