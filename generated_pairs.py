#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 07:06:05 2021

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

import multiprocessing    

from skimage.segmentation import watershed

from post_processing import erase_solitary

import time

import numpy as np
import warnings
import json

import bigfish
import bigfish.stack as stack

from scipy import ndimage as ndi
from scipy import ndimage, misc

from compute_spatial_state import generate_exels_cell_state_type










parser = argparse.ArgumentParser(description='test')

parser.add_argument('--path_to_exels_folder' ,type=str,
                    default = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/")
                       # default="/mnt/data3/tdefard/mic/to_take/exels_folders/one_cells_analysis/")
parser.add_argument('--path_to_take' ,type=str,
                     default = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/")
                        #default="/mnt/data3/tdefard/mic/to_take/")
parser.add_argument('--path_save' ,type=str,
                       default = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_0707/cell_pair/nuclei")
                      #default="/mnt/data3/tdefard/mic/to_take/exels_folders/one_cells_analysis_sp25/")


parser.add_argument("--probe_index", type=int, default = 0)

parser.add_argument("--couple", type=int, default = 1)

parser.add_argument("--state", type=int, default = 0)


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
"210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/",
]


path_load = args.path_to_exels_folder
path_to_take = args.path_to_take
path_save = args.path_save
import multiprocessing      
l_params = []
l_params = []

if args.couple:
   list_probes  =  [ [['Pdgfra'], ['Hhip']],
                              [ ['Pecam1'], ['Apln']], 
                                [['Pecam1'],  ['Ptprb']]] #[['Lamp3'], ['Pdgfra']],['C3ar1'],  ['Chil3']], 
   
   list_folder = ["210219_myo_fibros_y_macrophages/"]

   list_probes  =  [[['Pdgfra'], ['Hhip']]]
if args.state:
    list_probes  = [ ['Lamp3'],  ['Pdgfra'], ['Ptprb'],['Apln'], ['Chil3'], ['Fibin'], ['C3ar1'],  ['Hhip'],
           ['Pecam1'],  ['Cap', 'aCap', 'CEC'],
           ]#
    
    list_probes  = [  ['Cap', 'aCap', 'CEC'],
           ]
    list_probes = [[list_probes[i], ['Serpine1']] for i in range(len(list_probes))] + [[list_probes[i], ['Mki67']] for i in range(len(list_probes))] 
    print(list_probes)

for prb in list_probes :
        l_params.append([list_folder,
                    prb[0],
                    prb[1],
                    path_to_take,
                    path_save])
        
        
print("go")

for index in range(len(l_params)):
    print(index)
    generate_exels_cell_state_type(l_params[index])
#generate_exels_cell_state_type(l_params[args.probe_index])
#%%
import multiprocessing

number_processes = 2
pool = multiprocessing.Pool(number_processes)
results = pool.map_async(generate_exels_cell_state_type, l_params)
pool.close()
pool.join()


#%% rename dataframe column
path_to_exels = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_0507/cell_state_type/"

list_probes  = [[['Lamp3'], ['Serpine1']], 
                [['Pdgfra'], ['Serpine1']], 
                [['Ptprb'], ['Serpine1']],
                [['Apln'], ['Serpine1']], 
                [['Chil3'], ['Serpine1']],  
                [['Fibin'], ['Serpine1']], [['C3ar1'], ['Serpine1']],
                [['Hhip'], ['Serpine1']], [['Pecam1'], ['Serpine1']], 
                [['Cap', 'aCap', "CEC"], ['Serpine1']], [['Lamp3'], ['Mki67']],
                [['Pdgfra'], ['Mki67']], [['Ptprb'], ['Mki67']], [['Apln'], ['Mki67']],
                [['Chil3'], ['Mki67']], [['Fibin'], ['Mki67']], [['C3ar1'], ['Mki67']], 
                [['Hhip'], ['Mki67']], [['Pecam1'], ['Mki67']], [['Cap', 'aCap', 'CEC'], ['Mki67']]]

dico_text = {}
for prbs in list_probes:
        print(prbs)
        dataframe_per_files = pd.read_pickle(path_to_exels + prbs[0][0] +"_" +prbs[1][0] + ".pkl")
        
        
        columns_name = list(dataframe_per_files.columns)
        columns_name[6] = "nb_positive_to_cell_type_only"
        columns_name[7] = "nb_positive_to_cell_state_only"
        dataframe_per_files.columns = columns_name
        
        dataframe_per_files.to_pickle(path_to_exels  +  prbs[0][0] +"_" +prbs[1][0]  +".pkl")
        dataframe_per_files.to_excel(path_to_exels  +  prbs[0][0] +"_" +prbs[1][0]  +'.xls')