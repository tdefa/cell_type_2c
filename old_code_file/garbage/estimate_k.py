
import argparse

import cv2
import time
import os
from os import listdir
from os.path import isfile, join
import czifile as zis
from matplotlib import pyplot as plt
import tifffile
import numpy as np
import cellpose
from cellpose import models, io


import bigfish
import bigfish.detection as detection
import bigfish.plot as plot
import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.plot as plot

from post_processing import erase_solitary
from plot import  hsv_to_rgb

from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from utils import get_contours
from post_processing import erase_solitary, erase_small_nuclei
from tqdm import tqdm

from resnet_extractor import ResnetClassifier, ResnetClassifierOriginal, LeNet5, ResnetClassifierOriginal3
from skimage.transform import resize

from sklearn.cluster import OPTICS, cluster_optics_dbscan

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import argparse

import cv2
import time
import os
from os import listdir  
from os.path import isfile, join
import czifile as zis
from matplotlib import pyplot as plt
import tifffile
import numpy as np
import cellpose
from cellpose import models, io

from scipy.spatial import distance

import bigfish
import bigfish.detection as detection
from torchvision import datasets, models, transforms

from skimage.segmentation import watershed
from plot import mask_image_to_rgb

import alphashape

from descartes import PolygonPatch
#from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator
from collections import Counter
from scipy.spatial import Delaunay

from plot import mask_image_to_rgb

from matplotlib import pyplot as plt
import random
from skimage.io import imread, imsave
import numpy as np
from pathlib import Path
import json

import numpy as np
from geojson import Polygon as geojson_polygon
from shapely.geometry import Polygon as shapely_polygon
from geojson import Feature, FeatureCollection
from skimage import measure
from skimage.draw import  polygon
import tifffile
from skimage.segmentation import mark_boundaries, find_boundaries
from scipy import ndimage
from scipy.ndimage import gaussian_laplace
from geojson import dump
from geojson import dump
from geojson import Polygon as geojson_polygon
from shapely.geometry import Polygon as shapely_polygon
from geojson import Feature, FeatureCollection
from skimage import measure
from skimage.draw import  polygon

from geojson import Feature, Point
import matplotlib.path as mplPath

from spot_detection import computer_optics_cluster, generate_grid, cluster_over_nuclei_3D_convex_hull

def geojson_to_label(data_json, img_size, label = "nuclei",  binary_labeling=False):
    """
    Function reading a geojson and returning a label image array

    Args:
      data_json (dict): dictionary in the geojson format containing coordionate of object to label
      img_size (tuple int): size of the original labelled image
      binary_labeling (bool): if True it does not separate connected componnent and use 1 as label for all polygon
      if False the N objects are number 1,2,3,...N
    """

    def make_mask(roi_pos, img_size):
        rr, cc = polygon(roi_pos[:, 0], roi_pos[:, 1])
        # Make sure it's not outside
        rr[rr < 0] = 0
        rr[rr > img_size[0] - 1] = img_size[0] - 1
        cc[cc < 0] = 0
        cc[cc > img_size[1] - 1] = img_size[1] - 1
        # Generate mask
        mask = np.zeros(img_size, dtype=np.uint8)
        mask[rr, cc] = 1
        return mask

    mask_loop = np.zeros(img_size)
    color = 1
    for i in range(len(data_json['features'])):
        if data_json['features'][i]['properties']["label"]== label:
            try:
                reg_pos = np.squeeze(np.asarray(data_json['features'][i]['geometry']['coordinates']))
            except KeyError:
                pass  # GeometryCollection
            if len(reg_pos.shape) == 1:  # case where the label is a point
                reg_pos = np.array([reg_pos])
            reg_pos[:, [0, 1]] = reg_pos[:, [1, 0]]  # inverse coordinate from kaibu
            reg_pos[:, 0] = -1*reg_pos[:, 0]+img_size[0]
            if binary_labeling:
                mask_loop += make_mask(reg_pos, img_size)
            else:
                mask_loop += make_mask(reg_pos, img_size) * (color)  # N objectS are number 1,2,3,...N
            color += 1 
    return mask_loop
# -*- coding: utf-8 -*-



#%%
    
path_to_json1 = [
    "/home/tom/Bureau/annotation/rna_association/01_NI_Chil3-Cy3_Mki67-Cy5_02_mip/target_files_v1/annotation.json", 
"/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010_mip/target_files_v2/annotation.json",
 "/home/tom/Bureau/annotation/rna_association/11_NI_Cap-Cy3_Mki67-Cy5_002_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002.json",
"/home/tom/Bureau/annotation/rna_association/14_IR5M_Cap-Cy3_Serpine1-Cy5_011.json",
 "/home/tom/Bureau/annotation/rna_association/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06.json", 
"/home/tom/Bureau/annotation/rna_association/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10_mip/target_files_v3/annotation.json"]


onlyfiles1 = ["01_NI_Chil3-Cy3_Mki67-Cy5_02",
             "04_IR5M_Chil3-Cy3_Serpine1-Cy5_06", 
             "08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010",
             "11_NI_Cap-Cy3_Mki67-Cy5_002",
             "07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002", 
             "14_IR5M_Cap-Cy3_Serpine1-Cy5_011",
             "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06",
              "01_NI_Lamp3-Cy5_Pdgfra-Cy3_01",
             "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10"]


path_to_json2 = [
"/home/tom/Bureau/annotation/rna_association/02_IR5M_Chil3-Cy3_Mki67-Cy5_05_mip/target_files_v1/annotation.json", 
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_01/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_004_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_005_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_008/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_009_mip/target_files_v2/annotation_c.json"
]

onlyfiles2 = ["02_IR5M_Chil3-Cy3_Mki67-Cy5_05",
             "03_NI_Chil3-Cy3_Serpine1-Cy5_01", 
             "03_NI_Chil3-Cy3_Serpine1-Cy5_004",
             "04_IR5M_Chil3-Cy3_Serpine1-Cy5_02",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_005",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_008",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_009"
    ]

path_to_json3 = path_to_json1 + path_to_json2 
onlyfiles3  = onlyfiles1 + onlyfiles2

path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06.json", 
"/home/tom/Bureau/annotation/rna_association/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10_mip/target_files_v3/annotation.json"]

onlyfiles3  = ["03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06", "01_NI_Lamp3-Cy5_Pdgfra-Cy3_01",  "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10"]

path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_01/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_004_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/target_files_v1/annotation.json",]


onlyfiles3 = [ "04_IR5M_Chil3-Cy3_Serpine1-Cy5_06", "03_NI_Chil3-Cy3_Serpine1-Cy5_01", 
             "03_NI_Chil3-Cy3_Serpine1-Cy5_004",
             "04_IR5M_Chil3-Cy3_Serpine1-Cy5_02",]


#path_to_json3 = [ "/home/tom/Bureau/annotation/rna_association/14_IR5M_Cap-Cy3_Serpine1-Cy5_011.json"]
#onlyfiles3  = [ "14_IR5M_Cap-Cy3_Serpine1-Cy5_011"]
path_to_dapi = "/home/tom/Bureau/annotation/tiff_data2804/dapi/"
path_rna_647 = "/home/tom/Bureau/annotation/tiff_data2804/af647/"
path_rna_568 = "/home/tom/Bureau/annotation/tiff_data2804/af568/"

## state cell
onlyfiles3 = ["12_IR5M_Cap-Cy3_Mki67-Cy5_005",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_008",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_009"]


path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_005_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_008/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_009_mip/target_files_v2/annotation_c.json"
]

onlyfiles3 = [ "08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010",
 "07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002"]

path_to_json3 = [ "/home/tom/Bureau/annotation/rna_association/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010_mip/target_files_v2/annotation.json",
"/home/tom/Bureau/annotation/rna_association/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002.json"]


path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06.json", 
"/home/tom/Bureau/annotation/rna_association/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10_mip/target_files_v3/annotation.json"]

onlyfiles3  = ["03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06", "01_NI_Lamp3-Cy5_Pdgfra-Cy3_01",  "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10"]




onlyfiles3 = ["12_IR5M_Cap-Cy3_Mki67-Cy5_005",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_008",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_009"]


path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_005_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_008/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_009_mip/target_files_v2/annotation_c.json"
]


path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_01/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_004_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/target_files_v1/annotation.json",]


onlyfiles3 = [ "04_IR5M_Chil3-Cy3_Serpine1-Cy5_06", "03_NI_Chil3-Cy3_Serpine1-Cy5_01", 
             "03_NI_Chil3-Cy3_Serpine1-Cy5_004",
             "04_IR5M_Chil3-Cy3_Serpine1-Cy5_02",]

path_to_json1 = [
    "/home/tom/Bureau/annotation/rna_association/01_NI_Chil3-Cy3_Mki67-Cy5_02_mip/target_files_v1/annotation.json", 
"/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_06_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010_mip/target_files_v2/annotation.json",
 "/home/tom/Bureau/annotation/rna_association/11_NI_Cap-Cy3_Mki67-Cy5_002_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002.json",
"/home/tom/Bureau/annotation/rna_association/14_IR5M_Cap-Cy3_Serpine1-Cy5_011.json",
 "/home/tom/Bureau/annotation/rna_association/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06.json", 
"/home/tom/Bureau/annotation/rna_association/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10_mip/target_files_v3/annotation.json"]


onlyfiles1 = ["01_NI_Chil3-Cy3_Mki67-Cy5_02",
             "04_IR5M_Chil3-Cy3_Serpine1-Cy5_06", 
             "08_IR5M_Pdgfra-Cy3_Serpine1-Cy5_010",
             "11_NI_Cap-Cy3_Mki67-Cy5_002",
             "07_CtrlNI_Pdgfra-Cy3_Serpine1-Cy5_002", 
             "14_IR5M_Cap-Cy3_Serpine1-Cy5_011",
             "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06",
              "01_NI_Lamp3-Cy5_Pdgfra-Cy3_01",
             "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10"]


path_to_json2 = [
"/home/tom/Bureau/annotation/rna_association/02_IR5M_Chil3-Cy3_Mki67-Cy5_05_mip/target_files_v1/annotation.json", 
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_01/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_NI_Chil3-Cy3_Serpine1-Cy5_004_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/04_IR5M_Chil3-Cy3_Serpine1-Cy5_02/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_005_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_008/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/12_IR5M_Cap-Cy3_Mki67-Cy5_009_mip/target_files_v2/annotation_c.json"
]

onlyfiles2 = ["02_IR5M_Chil3-Cy3_Mki67-Cy5_05",
             "03_NI_Chil3-Cy3_Serpine1-Cy5_01", 
             "03_NI_Chil3-Cy3_Serpine1-Cy5_004",
             "04_IR5M_Chil3-Cy3_Serpine1-Cy5_02",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_005",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_008",
            "12_IR5M_Cap-Cy3_Mki67-Cy5_009"
    ]

path_to_json3 = path_to_json1 + path_to_json2 
onlyfiles3  = onlyfiles1 + onlyfiles2


path_to_json3 = ["/home/tom/Bureau/annotation/rna_association/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06/03_IRM_Lamp3_Cy3_Pdgfra-Cy5_06.json", 
"/home/tom/Bureau/annotation/rna_association/01_NI_Lamp3-Cy5_Pdgfra-Cy3_01_mip/target_files_v1/annotation.json",
"/home/tom/Bureau/annotation/rna_association/03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10_mip/target_files_v3/annotation.json"]

onlyfiles3  = ["03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_06", "01_NI_Lamp3-Cy5_Pdgfra-Cy3_01",  "03_IR5M_Lamp3-Cy5_Pdgfra-Cy3_10"]
#%%

def erase_overlapping_spot_k(spots_568, spots_647): #todo mettre la scale en parametre
    z_647 = np.array([s[0] for s in spots_647])
    x_647 = np.array([s[1] for s in spots_647])
    y_647 = np.array([s[2] for s in spots_647])
    z_568 = np.array([s[0] for s in spots_568])
    x_568 = np.array([s[1] for s in spots_568])
    y_568 = np.array([s[2] for s in spots_568])
    #r_dist = np.sqrt(np.square(z_647 * (300/103) - z_568.reshape(-1,1) * (300/103)) + np.square(np.square(x_647 - x_568.reshape(-1,1)) + np.square(y_647 - y_568.reshape(-1,1))))
    r_dist = np.sqrt(np.square(np.square(x_647 - x_568.reshape(-1,1)) + np.square(y_647 - y_568.reshape(-1,1))))
    min_d_spots_568 = []
    for s_b in range(len(spots_568)):
        min_d_spots_568.append([spots_568[s_b], r_dist[s_b].min()])
    
    min_d_spots_647 = []
    for s_b in range(len(spots_647)):
        min_d_spots_647.append([spots_647[s_b], r_dist[:,s_b].min()])

    return min_d_spots_568, min_d_spots_647

if __name__ == "__main__":
    dico_true_remove_spot = {}
    for kk in range(0,1):
        print(kk)
        dico_true_remove_spot[kk] = []
        spots_568_art_total = []
        spots_647_art_total = []
        spots_568_real_total = []
        spots_647_real_total = []
        for index in range(len(onlyfiles3)):
            print(onlyfiles3[index])
            json_file = path_to_json3[index]
            print(json_file)
            with open(str(json_file), encoding='utf-8-sig') as fh:
                data_json = json.load(fh)
                label_Cy3_noise = geojson_to_label(data_json, img_size=(1040,1388),label = "Cy3_noise", binary_labeling=False)
                label_Cy5_noise = geojson_to_label(data_json, img_size=(1040,1388),label = "Cy5_noise",
                                                  binary_labeling=False)
        
                label_Cy3 = geojson_to_label(data_json, img_size=(1040,1388), label = "Cy3", binary_labeling=False)
                label_Cy5 = geojson_to_label(data_json, img_size=(1040,1388), label = "Cy5",
                                                  binary_labeling=False)
        
                binnary_label_Cy5_noise = label_Cy5_noise >= 1
                binnary_label_Cy3_noise = label_Cy3_noise >= 1
                binnary_label_Cy5 = label_Cy5 >= 1
                binnary_label_Cy3 = label_Cy3 >= 1
                
            spots_647 = np.load("/home/tom/Bureau/annotation/tiff_data2804/spots_af647/" + onlyfiles3[index] + ".npy")
            spots_568 = np.load("/home/tom/Bureau/annotation/tiff_data2804/spots_af568/" + onlyfiles3[index] + ".npy")
            


            min_d_spots_568 , min_d_spots_647,  = erase_overlapping_spot_k(spots_568, spots_647)
            
            def get_real_art(spot, labels_real, labels_art):
                real = []
                art = []
                for s in spot:
                    if labels_real[tuple(s[0][1:])] and not labels_art[tuple(s[0][1:])]:
                        real.append(s)
                    if labels_art[tuple(s[0][1:])]:
                        art.append(s)
                return real, art
            spots_568_real, spots_568_art = get_real_art(min_d_spots_568, binnary_label_Cy3, binnary_label_Cy3_noise)
            spots_647_real, spots_647_art = get_real_art(min_d_spots_647, binnary_label_Cy5, binnary_label_Cy5_noise)
            spots_568_art_total += spots_568_art
            spots_647_art_total += spots_647_art
            spots_568_real_total += spots_568_real
            spots_647_real_total += spots_647_real
        dico_true_remove_spot[kk] += [min_d_spots_647, min_d_spots_568]
    
#%%
l568 = []
l647 = []       
for k in dico_true_remove_spot.keys():
    print(k)
    print(len(dico_true_remove_spot[k][0]))
    print(len(dico_true_remove_spot[k][1]))
    l568 += dico_true_remove_spot[k][0]
    l647 += dico_true_remove_spot[k][1]
    
    
    print()
    
    
    
print([])


#%%

spots_647_art_total


min3 = [s for s in spots_568_art_total if s[1] < 6]

min3_spots = [s[0] for s in spots_568_art_total if s[1] > 6 and s[1] < 10]



def min_distant(center, spots):
    avg = 0
    for s in spots:
        avg += np.linalg.norm((center-s)*np.array([300/103, 1, 1]))
        
    return avg/len(spots)

x_a = []
y_a = []
for i in range(11):
    min3_spots = [s[0] for s in spots_568_art_total if s[1] >= i and s[1] < i+1]
    if len(min3_spots) > 0:
        avg = min_distant(center = np.array([24, 520, 694]), spots = min3_spots)   
        print((i, len(min3_spots),avg))
        x_a.append(i)
        y_a.append(avg)
        





