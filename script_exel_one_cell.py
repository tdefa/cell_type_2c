#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 07:06:05 2021

@author: tom
"""
#%%

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
import time
import numpy as np
import warnings
import json


from compute_spatial_state import generate_exels_one_cell


if __name__ == "__main__":

    list_folder = [
        # "200828-NIvsIR5M/00_Capillary_EC/", #ok spot ok image, on image is wrong
        # "200828-NIvsIR5M/00_Large_Vessels/", #pb to rerun
        # "200828-NIvsIR5M/00_Macrophages/", #ok spot
        # "200908_CEC/",
        # "200908_fibrosis/",
        # "201030_fridyay/",
        # "201127_AM_fibro/", ##pb
        "210205_Prolicence/aCap_prolif/",
        "210205_Prolicence/aCap_senes/",
        # "210219_myo_fibros_y_macrophages/",
        # "210412_repeat_fibro/IR5M/",
        # "210412_repeat_fibro/NI/",
        # "210413_rep2/",
        # "210425_angiogenesis/",
        # "210426_repeat3/",
        # "210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/"
    ]

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--list_folder", nargs="+", default=list_folder,
                        help='list of folder to take into account')
    parser.add_argument("--gene_smfish", nargs="+",
                        default=['Cap', 'aCap', 'CEC', 'acap'],  help="write  the different name of a same probe ex Cap' aCap CEC acap")

    parser.add_argument("--path_to_main_folder", type=str,
                        default="/media/tom/Transcend/image_lustra0605/Images_Hugo/",
                        help="path to the folder that take contain all the folder")

    parser.add_argument("--path_to_save_excel", type=str,
                        default="/media/tom/Transcend/image_lustra0605/exelsonecell/",
                        help="path to the folder where result will be save")

    parser.add_argument("--dico_stat_name", type=str,
                        default="dico_seg1005.npy",
                        help="key world that is contain in name of the result file.")

    args = parser.parse_args()

    generate_exels_one_cell(list_folder=args.list_folder,
                            gene_smfish=args.gene_smfish,
                            path_to_take=args.path_to_main_folder,
                            path_save=args.path_to_save_excel,
                            dico_stat_name=args.dico_stat_name)






