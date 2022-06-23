# -*- coding: utf-8 -*-
import argparse

from dataset_classif_point import SpotDataset
from resnet_extractor import ResnetClassifier
from efficientnet_pytorch import EfficientNet
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

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, device="cpu"):
    print(device)
    since = time.time()
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        t = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_tp = 0
            running_p = 0
            running_tn = 0
            running_n = 0
            running_fp = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels["label"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #t = time.time()
                preds = preds.cpu()
                labels = labels.cpu()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_tp += sum(np.array(preds, dtype=bool) * np.array(labels.data, dtype=bool))
                running_p += sum(labels.data)
                running_tn += sum(~np.array(preds).astype(bool) * ~np.array(labels.data).astype(bool))
                running_n += sum(~np.array(labels.data).astype(bool))
                diff = np.array(labels.data) - np.array(preds)
                running_fp += len(np.where(diff == -1)[0])
                # print(t-time.time())
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("tpr %s " % str(running_tp/running_p))
            print("tnr %s" % str((running_tn/running_n)))
            print("precision %s" % str((running_tp/(running_tp+running_fp))))
            print("false negative rate %s" % str(running_fp))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print(time.time()-t)
            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    last_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, last_model_wts





def get_repartition(dataset):
    nb_zero = 0
    nb_one = 1

    for i in range(len(dataset)):
        if dataset[i][1]['label'] == 0:
            nb_zero += 1
        else:
            nb_one += 1
    return [nb_zero, nb_one]


# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-ptd', '--path_to_data',  type=str, default="/home/thomas/Bureau/phd/label_dataset", metavar='N', help='')
    parser.add_argument('-rz', '--resize',  type=int, default=644, metavar='N', help='')
    parser.add_argument('-bz', '--batch_size',  type=int, default=16, metavar='N', help='')
    parser.add_argument('--nb_workers',  type=int, default=4, metavar='N', help='')
    parser.add_argument('--fold',  type=int, default=0, metavar='N', help='')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    network = ResnetClassifier(torch.load('resnet18_pretrained.pt'))

    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.456, 0.406], [ 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Normalize([0.456, 0.406], [ 0.224, 0.225])
            ]),
        }

    optimizer_ft = optim.Adam(network.parameters(), lr=0.0001)  # optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.1)
    full_dataset1 = SpotDataset(args.path_to_data + "/mki67/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy5", data_transforms['val'], resize_size = args.resize)
    full_dataset2 = SpotDataset(args.path_to_data + "/mki67/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy5", data_transforms['val'], resize_size = args.resize)
    #full_dataset1 = SpotDataset(args.path_to_data + "/serpine/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy5", data_transforms['val'], resize_size = args.resize)
    full_dataset1 = SpotDataset(args.path_to_data + "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy5", data_transforms['val'], resize_size = args.resize)


    full_dataset2 = SpotDataset(args.path_to_data + "/lamp/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy5", data_transforms['val'], resize_size = args.resize)
    full_dataset3 = SpotDataset(args.path_to_data + "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy5", data_transforms['val'], resize_size = args.resize)
    #full_dataset2 = SpotDataset(args.path_to_data + "/serpine/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy5", data_transforms['val'], resize_size = args.resize)
    #full_dataset3 = SpotDataset( args.path_to_data + "/serpine/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy5", data_transforms['val'], resize_size = args.resize)
    #full_dataset4 = SpotDataset(args.path_to_data + "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy5", data_transforms['val'], resize_size = args.resize)

    """
    full_dataset9 , val_dataset = torch.utils.data.random_split(full_dataset9, [1000, len(full_dataset9)-1000], generator=torch.Generator().manual_seed(42))"""


    list_datasets = [full_dataset1, full_dataset2,  full_dataset3]

    if args.fold:
        range_max = len(list_datasets)
    else:
        range_max = 1

    for kfold in range(0, range_max):
        if args.fold:
            val_dataset = torch.utils.data.ConcatDataset([list_datasets[kfold]])
            train_dataset = torch.utils.data.ConcatDataset(list_datasets[:kfold] + list_datasets[kfold+1:])
        else:
            concat_dataset =  torch.utils.data.ConcatDataset(list_datasets)
            train_size = int(0.1 * len(concat_dataset))
            test_size = len(concat_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(concat_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(12))


        rep = get_repartition(train_dataset)
        weight = torch.tensor([rep[0] / sum(rep), rep[1] / sum(rep)]).to(device).double()
        criterion = nn.CrossEntropyLoss(weight=weight)



        train_dataset.transform = data_transforms['train']
        val_dataset.transform = data_transforms['val']
        dataloaders = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            shuffle=True, num_workers=args.nb_workers),
                       "val": torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.nb_workers)
                       }

        print(args.path_to_data)
        model, last_dict = train_model(network.double(), criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                                       dataloaders=dataloaders, num_epochs=65, device=device)
        torch.save(model.state_dict(), str(kfold) + "modelbest" + str(len(list_datasets)))
        torch.save(last_dict, str(kfold) + "modellast" + str(len(list_datasets )))
        print(kfold)
