# -*- coding: utf-8 -*-

import argparse
import os
import tifffile
from generate_graph import  label_with_empty, get_adjacent_list

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
        ### path to datset
    parser.add_argument('-ptt' ,"--path_to_take", type=str,
                            default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
                            help='path_to_czi folder')

    parser.add_argument('-pfn',"--folder_name", type=str, 
                            default= "210205_Prolicence/aCap_prolif/",
                            help='path_to_project')
    
    parser.add_argument("--threshold_empty", type=int, 
                            default= 40,
                            help='path_to_project')


    args = parser.parse_args()
    print(args)
    
    
    
    path_output_segmentaton = args.path_to_take + args.folder_name + "tiff_data/predicted_mask_dapi/"
    path_to_af568 = args.path_to_take + args.folder_name + "tiff_data/" + "af568/"
    if not os.path.exists(args.path_to_take + args.folder_name +  "adjacent_list/"):
        os.mkdir(args.path_to_take + args.folder_name + "adjacent_list/")
    onlyfiles = [f for f in os.listdir(path_output_segmentaton) if os.path.isfile(os.path.join(path_output_segmentaton, f)) and f[-1] == "f" ]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
        
    for f in onlyfiles:
        img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_"+ f )
        af568 = tifffile.imread(path_to_af568 + "AF568_"+ f )
        labels_with_empty = label_with_empty(img_dapi_mask, af568, threshold_percent = args.threshold_empty)
        print("labels_with_empty shape %s"  % str(labels_with_empty))
        adjacent_list, dico_ngb = get_adjacent_list(img_dapi_mask, labels_with_empty)
        np.save(args.path_to_take + args.folder_name + "adjacent_list/"+ "ad_list" + f, np.array(adjacent_list))
        np.save(args.path_to_take + args.folder_name+ "adjacent_list/" + "dico" + f, dico_ngb)
