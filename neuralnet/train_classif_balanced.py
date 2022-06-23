# -*- coding: utf-8 -*-
import argparse

from dataset_classif_point import SpotDataset
from resnet_extractor import ResnetClassifier, ResnetClassifierOriginal, LeNet5, ResnetClassifierOriginal3
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
from sampler import BalancedBatchSampler

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, device="cpu", early_stop = True):
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
            running_fn = 0
            total_length = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels["label"].to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # for:d
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
                total_length += len(preds)
                labels = labels.cpu()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_tp += sum(np.array(preds, dtype=bool) * np.array(labels.data, dtype=bool))
                running_p += sum(labels.data)
                running_tn += sum(~np.array(preds).astype(bool) * ~np.array(labels.data).astype(bool))
                running_n += sum(~np.array(labels.data).astype(bool))
                diff = np.array(labels.data) - np.array(preds)
                running_fp += len(np.where(diff == -1)[0])
                running_fn += len(np.where(diff == 1)[0])
                """print("positive %s" % str(sum(labels.data)))
                print("negative %s" % str(sum(~np.array(labels.data).astype(bool))))
                print("running_tp  %s" % str( sum(np.array(preds, dtype=bool) * np.array(labels.data, dtype=bool))))
                print("running_tn %s" % str(sum(~np.array(preds).astype(bool) * ~np.array(labels.data).astype(bool))))
                print("fp %s" % len(np.where(diff == -1)[0]))
                print("fn %s" % len(np.where(diff == 1)[0]))
                print()"""
                # print(t-time.time())
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / total_length
            epoch_acc = running_corrects.double() / total_length
            print("positive %s" % str(running_p))
            print("negative %s" % str(running_n))
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
    parser.add_argument('-rz', '--resize',  type=int, default=None, metavar='N', help='')
    parser.add_argument('-bz', '--batch_size',  type=int, default=16, metavar='N', help='')
    parser.add_argument('--epoch',  type=int, default=32, metavar='N', help='')

    parser.add_argument('--nb_workers',  type=int, default=4, metavar='N', help='')
    parser.add_argument('--fold',  type=int, default=1, metavar='N', help='')

    parser.add_argument('--model',  type=str, default="r18ori", metavar='N', help='r18, r18ori, r34, r34ori')
    parser.add_argument('--save_name',  type=str, default="default", metavar='N', help='')
    parser.add_argument('--dataset',  type=str, default="cy59", metavar='N', help='cy59, cy516, cy39,  cy38,cy37, cy311, chil3, pdgfra')
    parser.add_argument('--normalize',  type=int, default= 1, metavar='N', help='')
    parser.add_argument('--pretrained',  type=int, default= 1, metavar='N', help='')


    parser.add_argument('--sampling',  type=str, default= None, metavar='N', help='BalancedBatchSampler')

    parser.add_argument('--add_channel',  type=int, default= 0, metavar='N', help='BalancedBatchSampler')



    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    ################
    ##choose the model
    ##############
    if args.model == "r18":
        if args.pretrained:
            network = ResnetClassifier(torch.load('resnet18_pretrained.pt'))
        else:
            network = ResnetClassifier(models.resnet18())
    if args.model == "r18ori":
        if args.pretrained:
            network =ResnetClassifierOriginal(torch.load('resnet18_pretrained.pt'))
        else:
            network =ResnetClassifierOriginal(models.resnet18())

    if args.model == "r18ori3":
        if args.pretrained:
            network =ResnetClassifierOriginal3(torch.load('resnet18_pretrained.pt'))
        else:
            network = ResnetClassifierOriginal3(models.resnet18())
    if args.model == "r34":
        if args.pretrained:
            network = ResnetClassifier(torch.load('resnet34_pretrained.pt'))
        else:
            network = ResnetClassifierOriginal(models.resnet34())

    if args.model == "r34ori":
        if args.pretrained:
            network = ResnetClassifier(torch.load('resnet34_pretrained.pt'))
        else:
            network = ResnetClassifierOriginal(models.resnet34())
    if args.model == "lenet5":
            network = LeNet5()


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

    if args.dataset == 'cy59':
        name_list =  ["/mki67/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy5", "/mki67/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy5",
                                      "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy5", "/lamp/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy5",
                                      "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy5", "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy5",
                                      "/serpine/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy5", "/serpine/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy5",
                                      "/serpine/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy5"]

    if args.dataset == 'cy516':
            name_list =  ["/mki67/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy5",
            "/mki67/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy5",
            "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy5",
            "/lamp/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy5",
            "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy5",
            "/serpine/02_IR5M_Chil3-Cy3_Mki67-Cy5_05/Cy5",
            "/serpine/03_NI_Chil3-Cy3_Serpine1-Cy5_01/Cy5",
            "/serpine/03_NI_Chil3-Cy3_Serpine1-Cy5_004/Cy5",
            "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/Cy5",
            "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy5",
            "/serpine/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy5",
            "/serpine/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy5",
            "/serpine/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy5",]



    if args.dataset == 'cy39':
            name_list =  ["/chil3/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy3", "/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
                          "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3", "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy3",
                          "/pdgfra/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy3", "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy3",
                          "/pdgfra/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy3", "/pdgfra/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy3",
                          "/chil3/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy3"]

    if args.dataset == 'cy38':
            name_list =  [ "/chil3/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy3", "/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
                          "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3", "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy3",
                          "/pdgfra/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy3", "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy3",
                          "/pdgfra/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy3", "/pdgfra/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy3"]

    if args.dataset == 'cy37':
            name_list =  ["/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
                          "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3", "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy3",
                          "/pdgfra/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy3", "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy3",
                          "/pdgfra/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy3", "/pdgfra/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy3"]

    if args.dataset == 'cy311':
       name_list = ["/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy3",
        "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3",


        "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy3",
        "/pdgfra/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy3",
        "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy3",
        "/pdgfra/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy3",
        "/pdgfra/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy3"]


    if args.dataset == 'chil3':
            name_list =  [ "/chil3/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy3", #815 elements
                          "/chil3/02_IR5M_Chil3-Cy3_Mki67-Cy5_05/Cy3", #10 400
                          "/chil3/03_NI_Chil3-Cy3_Serpine1-Cy5_01/Cy3", #294
                          "/chil3/03_NI_Chil3-Cy3_Serpine1-Cy5_004/Cy3", # 1587
                          "/chil3/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/Cy3", # 5621
                          "/chil3/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy3", #11053
                          ]

    if args.dataset == 'pdgfra':
            name_list =  [ "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy3",
                            "/pdgfra/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy3",
                            "/pdgfra/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy3",
                            "/pdgfra/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy3",
                            "/pdgfra/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy3"]

    if args.dataset == 'cap':
        name_list = ["/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy3",
        "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3"]
        
    if args.dataset == 'cap_chill3':
        name_list = ["/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy3",
        "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3",
        "/chil3/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy3", #815 elements
        "/chil3/02_IR5M_Chil3-Cy3_Mki67-Cy5_05/Cy3", #10 400
        "/chil3/03_NI_Chil3-Cy3_Serpine1-Cy5_01/Cy3", #294
        "/chil3/03_NI_Chil3-Cy3_Serpine1-Cy5_004/Cy3", # 1587
        "/chil3/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/Cy3", # 5621
        "/chil3/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy3",]
    
    if args.dataset == 'chil3_pdgfra':
        name_list = ["/cap/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy3",
        "/cap/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy3",
        "/cap/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy3",]
        
    if args.dataset == 'mki67_serpine':
         name_list = ["/mki67/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy5",
            "/mki67/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy5",
            "/serpine/03_NI_Chil3-Cy3_Serpine1-Cy5_01/Cy5",
            "/serpine/03_NI_Chil3-Cy3_Serpine1-Cy5_004/Cy5",
            "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/Cy5",
            "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy5",
            "/serpine/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy5",
            "/serpine/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy5",
            "/serpine/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy5",]


    if args.dataset == 'lamp_serpine':
         name_list = ["/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy5",
            "/lamp/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy5",
            "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy5",
            "/serpine/03_NI_Chil3-Cy3_Serpine1-Cy5_01/Cy5",
            "/serpine/03_NI_Chil3-Cy3_Serpine1-Cy5_004/Cy5",
            "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/Cy5",
            "/serpine/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06/Cy5",
            "/serpine/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002/Cy5",
            "/serpine/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010/Cy5",
            "/serpine/14_IR5M_Cap-Cy3_Serpine1-Cy5_011/Cy5",]
         
    if args.dataset == 'lamp_mki67':
         name_list = ["/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06/Cy5",
            "/lamp/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01/Cy5",
            "/lamp/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10/Cy5",
            "/mki67/01_NI_Chil3-Cy3_Mki67-Cy5_02/Cy5",
            "/mki67/11_NI_Cap-Cy3_Mki67-Cy5_002/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_005/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_008/Cy5",
            "/mki67/12_IR5M_Cap-Cy3_Mki67-Cy5_009/Cy5",]


    list_datasets = []
    for name in name_list:
            list_datasets.append(SpotDataset(args.path_to_data + name, data_transforms['val'],
                                             resize_size = args.resize,
                                             normalize = args.normalize, add_channel = args.add_channel))


    if args.fold == 1:
        range_max = len(list_datasets)
    if args.fold == 0:
        range_max = 1

    for kfold in range(0, range_max):
        if args.fold==1:
            val_dataset = torch.utils.data.ConcatDataset([list_datasets[kfold]])
            train_dataset = torch.utils.data.ConcatDataset(list_datasets[:kfold] + list_datasets[kfold+1:])
        if args.fold==0:
            concat_dataset =  torch.utils.data.ConcatDataset(list_datasets)
            train_size = int(0.92 * len(concat_dataset))
            test_size = len(concat_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(concat_dataset,
                                            [train_size, test_size], generator=torch.Generator().manual_seed(12))



        #criterion = nn.CrossEntropyLoss(weight=weight)
        criterion = nn.CrossEntropyLoss()
        #https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703

        if args.sampling is None:
            shuffle = True
            sampler = None
            rep = get_repartition(train_dataset)
            weight = torch.tensor([rep[0] / sum(rep), rep[1] / sum(rep)]).to(device).double()
            criterion = nn.CrossEntropyLoss(weight=weight)

        if args.sampling == "BalancedBatchSampler":
            data_y = []
            for inp in train_dataset:
                data_y.append(inp[-1]['label'])
            sampler = BalancedBatchSampler(train_dataset, data_y)
            shuffle = False




        train_dataset.transform = data_transforms['train']
        val_dataset.transform = data_transforms['val']
        dataloaders = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            shuffle=shuffle, num_workers=args.nb_workers,
                                                            sampler = sampler),
                       "val": torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.nb_workers,
                                                          )
                       }
        print(args.path_to_data)
        model, last_dict = train_model(network.double(), criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler,
                                       dataloaders=dataloaders, num_epochs=args.epoch, device=device)
        torch.save(model.state_dict(), str(kfold) +name_list[kfold][-14:-4]+ "modelbest" + args.save_name + str(len(list_datasets)))
        torch.save(last_dict, str(kfold) + name_list[kfold][-14:-4]+ "modellast" + args.save_name +str(len(list_datasets)))
        print(kfold)




#%%


"""
nimages = 0
mean = 0.0
var = 0.0

for i_batch, batch_target in enumerate(dataloaders["train"]):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print(mean)
print(std)
"""