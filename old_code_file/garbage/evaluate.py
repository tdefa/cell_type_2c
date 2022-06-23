# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import tifffile
import scipy.stats
from post_processing import erase_solitary #, erase_small_nuclei

import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


import argparse
import os
from os import listdir
from os.path import isfile, join
import czifile as zis
from matplotlib import pyplot as plt
import tifffile
import numpy as np
import cellpose
from cellpose import models, io, plot

def plot_marked_volume(marked_image_volume, mask):
   for slice in range(len(marked_image_volume)):
       if np.count_nonzero(mask[slice,:,:]):
           plt.figure(figsize=(10,10))
           edges_pz = mark_boundaries(marked_image_volume[slice,:,:], mask[slice].astype(np.int),
                                                color=(1,0,0), mode='thin')
           plt.imshow(edges_pz)
           plt.title('slice ' + str(slice))
           plt.show()


def evaluate_iou(path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14",
                 path_prediction = "/home/thomas/Bureau/phd/first_one/tiff_data/predicted_mask_dapi_st04", d3 = True, iou_thresh = 0.5, erase_so = True):


    path_label = Path(path_label)
    path_prediction = Path(path_prediction)
    ap_list = []
    ac_list =[]
    for image_files_label in path_label.glob("*/"):
       # print(image_files_label)
        image_label = imread(image_files_label)
        try:
            slice_label = int(list(image_files_label.parts)[-1][-6:-4])
            image_name = str(list(image_files_label.parts)[-1][:-9])
        except :
            slice_label = int(list(image_files_label.parts)[-1][-5:-4])
            image_name = str(list(image_files_label.parts)[-1][:-8])
        if erase_so:
            image_pred =  erase_solitary(tifffile.imread(str(path_prediction) +"/dapi_maskdapi_" +  image_name  + ".tiff"))
        else:
            image_pred =  tifffile.imread(str(path_prediction) +"/dapi_maskdapi_" +  image_name  + ".tiff")
        cell_label = np.unique(image_label)
        cell_pred = np.unique(image_pred[slice_label])
        dico_ap = {}
        for c in cell_label:
            current_cell = (image_label == c).astype(bool)
            for pred in cell_pred:
                overlap = current_cell * (image_pred[slice_label] == pred).astype(bool) # Logical AND
                union = current_cell +  (image_pred[slice_label] == pred).astype(bool)# Logical OR
                if overlap.sum()/float(union.sum()) > iou_thresh:
                    dico_ap[c] = pred
        tp = len(dico_ap)
        fp = len(cell_pred)  - len(dico_ap.values()) #len([cell_pred[i] for i in range(len(cell_pred)) if cell_pred[i] not in dico_ap.values()])
        fn = len(cell_label) - len(dico_ap.keys()) #len([cell_label[i] for i in range(len(cell_label)) if cell_label[i] not in dico_ap.keys()])
    
        ap = tp / (tp+fp+fn)
        ap_list.append(ap)
    return ap_list



def compute_ap(image_pred, image_label, iou_thresh = 0.5):
    cell_label = np.unique(image_label)
    cell_pred = np.unique(image_pred[slice_label])
    dico_ap = {}
    for c in cell_label:
        current_cell = (image_label == c).astype(bool)
        for pred in cell_pred:
            overlap = current_cell * (image_pred[slice_label] == pred).astype(bool) # Logical AND
            union = current_cell +  (image_pred[slice_label] == pred).astype(bool)# Logical OR
            if overlap.sum()/float(union.sum()) > iou_thresh:
                dico_ap[c] = pred
    tp = len(dico_ap)
    fp = len(cell_pred)  - len(dico_ap.values()) #len([cell_pred[i] for i in range(len(cell_pred)) if cell_pred[i] not in dico_ap.values()])
    fn = len(cell_label) - len(dico_ap.keys()) #len([cell_label[i] for i in range(len(cell_label)) if cell_label[i] not in dico_ap.keys()])

    ap = tp / (tp+fp+fn)
    return ap



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def generate_datset():
    path_save = Path(r'/home/thomas/Bureau/phd/kaibu_data/dandra_3d_14')   # Folder to store data
    path_new_dataset = Path(r'/home/thomas/Bureau/phd/label_dataset/sandra_3d_v0')
    for folder in path_save.glob("*/"):
        print(folder)
        image = folder / "target_files_v0/annotation.png"
        if not image.is_file():
            print("t")
            image = folder / "target_files_v0/annotation.png"
        image = imread(image)

        imsave(str(path_new_dataset / list(folder.parts)[-1]) + ".png", image)

    plt.imshow(image)



def from_labe_to_training(): #from label dataset to dataset for training;
    path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14"
    path_input = "/home/thomas/Bureau/phd/first_one/tiff_data/dapi"
    path_label = Path(path_label)
    path_input = Path(path_input)


#%%
import os

import numpy as np
from deepcell.applications import NuclearSegmentation

import imageio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from deepcell.applications import NuclearSegmentation
from deepcell.applications import CytoplasmSegmentation






app = NuclearSegmentation()
if __name__ == "__main__":
    path_label = "/home/tom/Bureau/annotation/dapi_annotation/dandra_3d_14"
    path_input = "/home/tom/Bureau/annotation/dapi_annotation/dapi_dandra3D_14/"
    path_to_save = "/home/thomas/Bureau/phd/label_dataset/with_name/"
    path_label = Path(path_label)
    
    
    
    d3 = False
    iou_thresh = 0.5
    erase_so = True
    
    parser = argparse.ArgumentParser(description='test')
    ### path to dapi tiff datset
    parser.add_argument('-pi',"--path_input", type=str, default= "/home/thomas/Bureau/phd/tiff_data/dapi/" , help='')
    parser.add_argument('-po',"--path_output", type=str, default= "/home/thomas/Bureau/phd/tiff_data/" , help='')

    ###cellpose arg
    parser.add_argument('-d',"--diameter", type=float, default= None, help='')
    parser.add_argument('-ft',"--flow_threshold", type=float, default=0.4, help='')
    parser.add_argument('-d3',"--do_3D", type=bool, default= False, help='')
    parser.add_argument('-m',"--mip", type=bool, default=True, help='')
    parser.add_argument('-st',"--stitch_threshold", type=float, default= 0, help='')
    parser.add_argument('-mode',"--mode", type=str, default='nuclei', help='')


    args = parser.parse_args()
    
    ###parameter
    dico_param = {}
    dico_param["diameter"]  = args.diameter
    dico_param["flow_threshold"] = args.flow_threshold
    dico_param["do_3D"] = args.do_3D
    dico_param["mip"] = args.mip
    dico_param["projected_focused"] = False
    dico_param["stitch_threshold"] = args.stitch_threshold
    model = models.Cellpose(gpu=False, model_type=args.mode)

    
    
    i = 0
    ap_cellpose = []
    ap_mesmer = []
    for image_files_label in path_label.glob("*/"):
        print(i)
        image_label = imread(image_files_label)
        try:
            slice_label = int(list(image_files_label.parts)[-1][-6:-4])
            image_name = str(list(image_files_label.parts)[-1][:-9])
        except :
            slice_label = int(list(image_files_label.parts)[-1][-5:-4])
            image_name = str(list(image_files_label.parts)[-1][:-8])
        print(slice_label)
        print(image_name)

        image_input =  tifffile.imread(str(path_input +"/dapi_" +  image_name  + ".tiff"))

        masks, flows, styles, diams = model.eval(image_input[slice_label], diameter=dico_param["diameter"], channels=[0,0],
                                                flow_threshold=dico_param["flow_threshold"], do_3D= dico_param["do_3D"],
                                                stitch_threshold = dico_param["stitch_threshold"])
        
        plt.imshow(image_input[slice_label], cmap = 'Blues')
        plt.title('dapi')
        plt.show()
        image_input = image_input.reshape((54, 1040, 1388,1))
        y_pred = app.predict(image_input[slice_label].reshape((1, 1040, 1388,1)))

        plt.imshow(image_label)
        plt.title('image_label')
        plt.show()
        plt.imshow(masks)
        plt.title('cellpose')
        plt.show()
        
        plt.imshow(y_pred[0,:,:,0])
        plt.title('mesmer')
        plt.show()
        ap_mesmer.append(compute_ap(image_pred = y_pred[0,:,:,0], image_label =image_label , iou_thresh = 0.5))
        print(ap_mesmer)
        ap_cellpose.append(compute_ap(image_pred = masks, image_label = image_label, iou_thresh = 0.5))
        print(ap_cellpose)
                

### get the corresponding images

#%%

    path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14"
    path_prediction = "/home/thomas/Bureau/phd/first_one/tiff_data/predicted_mask_dapi_st04"
    d3 = True
    iou_thresh = 0.5
    erase_so = 


    path_label = Path(path_label)
    path_prediction = Path(path_prediction)
    ap_list = []
    ac_list =[]
    for image_files_label in path_label.glob("*/"):
       # print(image_files_label)
        image_label = imread(image_files_label)
        try:
            slice_label = int(list(image_files_label.parts)[-1][-6:-4])
            image_name = str(list(image_files_label.parts)[-1][:-9])
        except :
            slice_label = int(list(image_files_label.parts)[-1][-5:-4])
            image_name = str(list(image_files_label.parts)[-1][:-8])
        if erase_so:
            image_pred =  erase_solitary(tifffile.imread(str(path_prediction) +"/dapi_maskdapi_" +  image_name  + ".tiff"))
        else:
            image_pred =  tifffile.imread(str(path_prediction) +"/dapi_maskdapi_" +  image_name  + ".tiff")
        cell_label = np.unique(image_label)
        cell_pred = np.unique(image_pred[slice_label])
        dico_ap = {}

        for c in cell_label:
            current_cell = (image_label == c).astype(bool)
            i = 0

            for pred in cell_pred:
                overlap = current_cell * (image_pred[slice_label] == i).astype(bool) # Logical AND
                union = current_cell +  (image_pred[slice_label] == i).astype(bool)# Logical OR
                if overlap.sum()/float(union.sum()) > iou_thresh:
                    dico_ap[c] = pred
                i += 1
        tp = len(dico_ap)
        fp = len(cell_pred)  - len(dico_ap.values()) #len([cell_pred[i] for i in range(len(cell_pred)) if cell_pred[i] not in dico_ap.values()])
        fn = len(cell_label) - len(dico_ap.keys()) #len([cell_label[i] for i in range(len(cell_label)) if cell_label[i] not in dico_ap.keys()])
    
        ap = tp / (tp+fp+fn)
        ap_list.append(ap)
        print(ap_list)
        print(sum(ap_list)/len(ap_list))

#%%




if __name__ == "__main__":
    
    path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14"
    path_prediction = "/home/thomas/Bureau/phd/first_one/tiff_data/predicted_mask_dapi_st04"
    d3 = True
    iou_thresh = 0.5
    erase_so = True
    
    parser = argparse.ArgumentParser(description='test')
    ### path to dapi tiff datset
    parser.add_argument('-pi',"--path_input", type=str, default= "/home/thomas/Bureau/phd/tiff_data/dapi/" , help='')
    parser.add_argument('-po',"--path_output", type=str, default= "/home/thomas/Bureau/phd/tiff_data/" , help='')

    ###cellpose arg
    parser.add_argument('-d',"--diameter", type=float, default= None, help='')
    parser.add_argument('-ft',"--flow_threshold", type=float, default=0.4, help='')
    parser.add_argument('-d3',"--do_3D", type=bool, default= False, help='')
    parser.add_argument('-m',"--mip", type=bool, default=True, help='')
    parser.add_argument('-st',"--stitch_threshold", type=float, default= 0, help='')
    parser.add_argument('-mode',"--mode", type=str, default='nuclei', help='')


    args = parser.parse_args()
    
    ###parameter
    dico_param = {}
    dico_param["diameter"]  = args.diameter
    dico_param["flow_threshold"] = args.flow_threshold
    dico_param["do_3D"] = args.do_3D
    dico_param["mip"] = args.mip
    dico_param["projected_focused"] = False
    dico_param["stitch_threshold"] = args.stitch_threshold
    model = models.Cellpose(gpu=False, model_type=args.mode)

    #folder_name = predicted_mask_dapi + args.mode +'_' +str(args.stitch_threshold) + '_' + str(args.flow_threshold)
    list_mask = []
    list_label = []
    path_label = Path(path_label)
    ap_list = []

    for image_files_label in path_label.glob("*/"):
        print(image_files_label)
        image_label = imread(image_files_label)
        try:
            slice_label = int(list(image_files_label.parts)[-1][-6:-4])
            image_name = str(list(image_files_label.parts)[-1][:-9])
        except :
            slice_label = int(list(image_files_label.parts)[-1][-5:-4])
            image_name = str(list(image_files_label.parts)[-1][:-8])
        image_input =  tifffile.imread(str(args.path_input +"/dapi_" +  image_name  + ".tiff"))
        
        masks, flows, styles, diams = model.eval(image_input[slice_label], diameter=dico_param["diameter"], channels=[0,0],
                                                flow_threshold=dico_param["flow_threshold"], do_3D= dico_param["do_3D"],
                                                stitch_threshold = dico_param["stitch_threshold"])
        masks = np.array(masks)
        list_mask.append(masks)
        list_label.append(image_label)
        
#%%
    ap_list = []
    for i in range(len(list_mask)):
        image_pred = list_mask[i]
        image_label = list_label[i]
        cell_label = np.unique(image_label)
        cell_pred = np.unique(image_pred)
        dico_ap = {}
        for c in cell_label:
            current_cell = (image_label == c).astype(bool)
            i=0
            for pred in cell_pred:
                overlap = current_cell * (image_pred == pred).astype(bool) # Logical AND
                union = current_cell +  (image_pred  == pred).astype(bool)# Logical OR
                if overlap.sum()/float(union.sum()) > iou_thresh:
                    dico_ap[c] = pred
                i += 1
        tp = len(dico_ap)
        fp = len(cell_pred)  - len(dico_ap.values()) #len([cell_pred[i] for i in range(len(cell_pred)) if cell_pred[i] not in dico_ap.values()])
        fn = len(cell_label) - len(dico_ap.keys()) #len([cell_label[i] for i in range(len(cell_label)) if cell_label[i] not in dico_ap.keys()])
    
        ap = tp / (tp+fp+fn)
        ap_list.append(ap)
        print(ap)
    print(ap_list)
    print(sum(ap_list)/len(ap_list))

#

#
    
    

