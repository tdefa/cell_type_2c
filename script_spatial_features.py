#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:12:00 2021

@author: tom
"""
import time
import pandas as pd
import argparse
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import tifffile
import numpy as np



from skimage.segmentation import watershed

from post_processing import erase_solitary

import time

import numpy as np
import warnings
import json

from scipy import ndimage as ndi
from scipy import ndimage, misc
###########Function area
#from tvtk.common import configure_input
import numpy as np
#from tvtk.api import tvtk
#from tvtk.common import configure_input
from scipy.spatial.distance import pdist, squareform
from skimage.segmentation import watershed
from scipy import ndimage as ndi

from tqdm import tqdm

from pathlib import Path 

from compute_spatial_state import get_knn_ratio, get_dye, count_positive_cell
import multiprocessing


#%%



parser = argparse.ArgumentParser(description='test')

parser.add_argument('--path_to_tiff' ,type=str,
                        default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/")


parser.add_argument("--probe_index1", type=int, default = 0)
parser.add_argument("--probe_index2", type=int, default = -1)

parser.add_argument("--nb_process", type=int, default = 1)
args = parser.parse_args()
print(args)
        
list_folder = [
    "200828-NIvsIR5M/00_Capillary_EC/", #ok spot ok image, on image is wrong
    "200828-NIvsIR5M/00_Large_Vessels/", #pb to rerun
    "200828-NIvsIR5M/00_Macrophages/", #ok spot
    "200908_CEC/", 
    "200908_fibrosis/",
    "201030_fridyay/",
    "201127_AM_fibro/", ##pb
    "210205_Prolicence/aCap_prolif/",
    "210205_Prolicence/aCap_senes/",
    "210219_myo_fibros_y_macrophages/",
    "210412_repeat_fibro/IR5M/",
    "210412_repeat_fibro/NI/",
    "210413_rep2/",
    "210425_angiogenesis/",
    "210426_repeat3/",
    ]
list_folder = list_folder[args.probe_index1: args.probe_index2]
def save_mask(l_params):
    t = time.time()
    print(l_params)
    folder_name, f = l_params[0], l_params[1]
    local_path = args.path_to_tiff
    path_output_segmentaton = local_path + folder_name + "tiff_data/" + "predicted_mask_dapi/"

    disk_path = local_path
    path_to_watersheld = disk_path + folder_name + "tiff_data/watersheld/"
    img_dapi_mask = tifffile.imread(path_output_segmentaton  + f)
    img_dapi_mask = erase_solitary(img_dapi_mask)
    inverted_mask = np.ones(img_dapi_mask.shape) - (img_dapi_mask != 0).astype(np.int)
    if len(img_dapi_mask.shape) == 3:
        distance = ndi.distance_transform_edt(inverted_mask, 
                                              sampling = [300, 103, 103])
    else:
        distance = ndi.distance_transform_edt(inverted_mask)  # compute distance map to border
    t = time.time()
    labels = watershed(distance, img_dapi_mask)
    np.save(path_to_watersheld +"watersheld"+f, labels)
    print(time.time() -t)
    
    
        
    

for folder_name in list_folder:
    print(folder_name)
    local_path = args.path_to_tiff
    path_output_segmentaton = local_path + folder_name + "tiff_data/" + "predicted_mask_dapi/"
    
    disk_path = args.path_to_tiff
    path_to_watersheld = disk_path + folder_name + "tiff_data/watersheld/"
    if not os.path.exists(path_to_watersheld):
         os.mkdir(path_to_watersheld )
    onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
    l_params = []
    for f in onlyfiles:
        l_params.append([folder_name, f])
    print('no parralele')
    for l in l_params:
        save_mask(l)
    
        




