#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:27:55 2021

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

from compute_spatial_state import update_dataframe_couple_cell



parser = argparse.ArgumentParser(description='test')

parser.add_argument('--path_to_exels_folder' ,type=str,
                        default="/mnt/data3/tdefard/mic/to_take/exels_folders/cell_state_cell_type/")
parser.add_argument('--path_to_take' ,type=str,
                        default="/mnt/data3/tdefard/mic/to_take/")
parser.add_argument('--path_save' ,type=str,
                        default="/mnt/data3/tdefard/mic/to_take/exels_folders/cell_type_couple_sp/")


parser.add_argument("--probe_index", type=int, default = 0)

parser.add_argument("--couple", type=int, default = 0)

parser.add_argument("--state", type=int, default = 1)


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


path_load = args.path_to_exels_folder
path_to_take = args.path_to_take
path_save = args.path_save
import multiprocessing      
l_params = []
l_params = []

if args.couple:
   list_probes  =  [[['C3ar1'],  ['Chil3']], [['Lamp3'], ['Pdgfra']], [['Pdgfra'], ['Hhip']],
                              [ ['Pecam1'], ['Apln']], 
                                [['Pecam1'],  ['Ptprb']]] 
if args.state:
    list_probes  = [ ['Lamp3'],  ['Pdgfra'], ['Ptprb'],['Apln'], ['Chil3'], ['CEC'], ['Fibin'], ['C3ar1'],  ['Hhip'],
           ['Pecam1'],  ['Cap', 'aCap'],
           ]#
    list_probes = [[list_probes[i], ['Serpine1']] for i in range(len(list_probes))] + [[list_probes[i], ['Mki67']] for i in range(len(list_probes))] 
    print(list_probes)

for prb in list_probes :
        l_params.append([list_folder,
                    prb[0],
                    prb[1],
                    path_load, 
                    path_to_take,
                    path_save])

update_dataframe_couple_cell(l_params[args.probe_index])