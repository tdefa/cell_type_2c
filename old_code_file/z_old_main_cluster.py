# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Mar 31 11:27:43 2021

@author: thomas
"""
import time
import argparse
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import tifffile
from utils.czi_to_tiff import preprare_tiff

#import cellpose
from cellpose import models

from run_seg import segment_nuclei

from spots.spot_detection import computer_optics_cluster
from spots.spot_detection import cluster_over_nuclei_3D_convex_hull, spot_detection_for_clustering, mask_image_to_rgb2D_from_list
#from spots.spot_detection import mask_image_to_rgb2D_from_list_green_cy3_red_cy5_both_blue_grey
from spots.plot import mask_image_to_rgb

from spots.post_processing import erase_solitary

from spots.erase_overlapping_spot import erase_point_in_cluster_2Dalphashape, erase_overlapping_spot

#from compute_spatial_state import compute_average_size
import numpy as np
import warnings
import multiprocessing
warnings.filterwarnings("ignore")

     



#%%                  
list_folder = [
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/", #ok
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/", #ok
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",#okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/", #okd
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/", #ok
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/", #ok
#"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3"
]

def main(list_folder):
    dico_cy3 = {"02_NI1230_Lamp3-Cy5_Pdgfra-Cy3_04.tiff" : 45}
    dico_cy5 = {"02_NI1230_Lamp3-Cy5_Pdgfra-Cy3_08.tiff" : 8,
                 "01_IR5M1236_Lamp3-Cy5_Pdgfra-Cy5_04.tiff" : 7,
                 "05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_08.tiff" : 9,
                 "05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_01.tiff" : 9,}

    dico_param_probes = {"Lamp3": (32, 0.42),
                  "Pdgfra" : (35, 0.42),
                  "Chil3": (20, 0.55),
                  'Cap': (35, 0.30),
                  'aCap': (35, 0.30),
                  "Ptprb": (27, 0.45),
                  "Fibin": (27, 0.40),
                 'C3ar1': (35, 0.45),
                 'Hhip': (35, 0.25),
                 'Mki67': (40, 0.30),
                  "Serpine1": (40, 0.50),
                  "Apln": (30, 0.40),
                  "Pecam1": (30, 0.40),
                  "CEC": (35, 0.30),
                  }




        
    #list_folder = [x[0] for x in os.walk("/mnt/data3/tdefard/mic/to_take")]
    #list_folder.reverse()
    list_folder_project  = list_folder
    for folder_index in range(len(list_folder)):

        print(list_folder)
        print(list_folder[folder_index])

    
        folder = list_folder[folder_index]
        parser = argparse.ArgumentParser(description='test')
        ### path to datset
        parser.add_argument('-ptz',"--path_to_czi_folder", type=str,
                            default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/"+list_folder[folder_index], help='path_to_czi folder')
        
        parser.add_argument('-ptp',"--path_to_project", type=str, 
                            default= "/home/tom/Bureau/annotation/cell_type_annotation/to_take/" + list_folder_project[folder_index],
                            help='path_to_project')
    
        ###cellpose arg
        parser.add_argument('-d',"--diameter", type=float, default= None, help='')
        parser.add_argument('-ft',"--flow_threshold", type=float, default=0.55, help='')
        parser.add_argument('-d3',"--do_3D", type=bool, default= False, help='')
        parser.add_argument('-m',"--mip", type=bool, default=False, help='')
        parser.add_argument('-st',"--stitch_threshold", type=float, default= 0.4, help='')
        parser.add_argument('-er',"--erase_solitary", type=int, default=1, help='')
        
        parser.add_argument('-prczi',"--prepare_czi", type=int, default=0, help='')
        parser.add_argument('-sg',"--segmentation", type=int, default=0, help='')
        parser.add_argument("--spot_detection", type=int, default=0, help='')
        parser.add_argument("--save_plot", type=int, default=1, help='')
        parser.add_argument("--clustering", type=int, default=1, help='')
        
        parser.add_argument("--epsi_cluster_cy3",  default="é", help='')
        parser.add_argument("--epsi_cluster_cy5", default="e", help='')
        
        parser.add_argument("--epsi_alphashape_cy3", type=int, default=25, help='')
        parser.add_argument("--epsi_alphashape_cy5", type=int, default=25, help='')
    
        parser.add_argument("--overlapping_cy3", default="e", help='')
        parser.add_argument("--overlapping_cy5",  default="e", help='')
        
        parser.add_argument("--remove_overlaping", type=int, default=1, help='')
    
        parser.add_argument("--kk_568", type=int, default = 3)    
        parser.add_argument("--kk_647", type=int, default = 3)    
    
        args = parser.parse_args()
        print(args)
        
            
        if not os.path.exists(args.path_to_czi_folder):
            os.mkdir(args.path_to_czi_folder)
            
        if not os.path.exists(args.path_to_czi_folder + "tiff_data/"):
            os.mkdir(args.path_to_czi_folder + "tiff_data/")
        if not os.path.exists(args.path_to_czi_folder + "tiff_data/" + "dapi/"):
             os.mkdir(args.path_to_czi_folder+ "tiff_data/" + "dapi/")
        if not os.path.exists(args.path_to_czi_folder+ "tiff_data/" + "af568/"):
             os.mkdir(args.path_to_czi_folder+ "tiff_data/" + "af568/")
        if not os.path.exists(args.path_to_czi_folder + "tiff_data/" + "af647/"):
             os.mkdir(args.path_to_czi_folder + "tiff_data/" + "af647/")
        
        path_to_czi = args.path_to_czi_folder
        path_to_dapi = args.path_to_czi_folder + "tiff_data/" + "dapi/"
        path_to_af647 = args.path_to_czi_folder + "tiff_data/" + "af647/"
        path_to_af568 = args.path_to_czi_folder + "tiff_data/" + "af568/"
        
        
        path_output_segmentaton = args.path_to_czi_folder + "tiff_data/" + "predicted_mask_dapi/"

        #################
        # CZI TO tiff
        #################
        if args.prepare_czi:
            try:
                preprare_tiff(path_to_czi, path_to_dapi, path_to_af647, path_to_af568)
            except Exception as e:
                print(e)
                
        if not os.path.exists(path_output_segmentaton):
             os.mkdir(path_output_segmentaton)
        onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
        onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
        

                
        
        
        ################
        # Nuclei segmentation
        ################
        
        
    
        
        if args.segmentation:
            model = models.Cellpose(gpu=True, model_type='nuclei')
            # ##parameter
            dico_param = {}
            dico_param["diameter"]  = args.diameter
            dico_param["flow_threshold"] = args.flow_threshold
            dico_param["do_3D"] = args.do_3D
            dico_param["mip"] = args.mip
            dico_param["projected_focused"] = False 
            dico_param["stitch_threshold"] = args.stitch_threshold
            segment_nuclei(path_to_dapi  , path_output_segmentaton, dico_param, model)   
        
        assert len(onlyfiles) <= len([f for f in listdir(path_to_dapi ) if isfile(join(path_to_dapi, f)) and f[-1] == "f" ])

        if args.spot_detection:
             print("spotdetection")
             if not os.path.exists(args.path_to_project + "detected_spot_3d"+"/"):
                  os.mkdir(args.path_to_project + "detected_spot_3d"+"/")
         
             dico_threshold = spot_detection_for_clustering(sigma = (1.25, 1.25, 1.25), float_out= False,
                                           rna_path = [path_to_af568+'AF568_'],
                                           path_output_segmentaton = path_output_segmentaton,
                                      threshold_input = dico_cy3,
                                      output_file = args.path_to_project + "detected_spot_3d"+"/",)
             np.save(args.path_to_project + 'AF568.npy', dico_threshold)
             print(dico_threshold)
             dico_threshold = spot_detection_for_clustering(sigma = (1.35, 1.35, 1.35), float_out= True,
                                           rna_path = [path_to_af647+'AF647_'],
                                           path_output_segmentaton = path_output_segmentaton,
                                      threshold_input = dico_cy5,
                                      output_file = args.path_to_project + "detected_spot_3d"+"/",)
             np.save(args.path_to_project +'AF647.npy', dico_threshold)
             print(dico_threshold)
    
            
        
        if args.save_plot:
            print("ploting")
            if not os.path.exists( args.path_to_project + "plot_clustering_artifact_sup/"):
                 os.mkdir(args.path_to_project + "plot_clustering_artifact_sup/")
    
    
            dico_stat = {}
            list_nuclei_criteria = []
            nb_fileee = 0
            t = time.time()
            #try:
              #  dico_stat = np.load(args.path_to_project + "dico_stat_2106.npy", allow_pickle =True).item()
            #except :
             #   print("édico not found")
            dico_stat = np.load(args.path_to_project + "dico_stat_2106.npy", allow_pickle =True).item()
            print(onlyfiles)
            for f in onlyfiles[:]:
                if f  in list(dico_stat.keys()):
                    print("alredy in " + f)
                    continue
                print(list_folder[folder_index])
                print("not alredy in " + f)
                print(nb_fileee)
                print(time.time() -t)
                print(f[:-5])
    
                t = time.time()
                nb_fileee += 1
    
                
                
                #### set clustering parameters  ###
                for probe_name in dico_param_probes.keys():
                    if probe_name +'-Cy3' in f:
                        args.epsi_cluster_cy3 = dico_param_probes[probe_name][0]
                        args.overlapping_cy3 = dico_param_probes[probe_name][1]
                        
                    if probe_name +'-Cy5' in f:
                        args.epsi_cluster_cy5 = dico_param_probes[probe_name][0]
                        args.overlapping_cy5 = dico_param_probes[probe_name][1]
                        
                    if probe_name +'Cy3' in f:
                        args.epsi_cluster_cy3 = dico_param_probes[probe_name][0]
                        args.overlapping_cy3 = dico_param_probes[probe_name][1]
                        
                    if probe_name +'Cy5' in f:
                        args.epsi_cluster_cy5 = dico_param_probes[probe_name][0]
                        args.overlapping_cy5 = dico_param_probes[probe_name][1]
                    
                ####
                if not os.path.exists( args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"):
                    os.mkdir(args.path_to_project + "plot_clustering_artifact_sup/"+ f[:-5] +"/")
                print(f)
                img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f)
                img_dapi_mask = erase_solitary(img_dapi_mask)
                """try:
                    img_dapi_mask = tifffile.imread(path_output_segmentaton +"erase_solitary/"+  "dapi_maskdapi_"+ f)
                except:
                    img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f)
                    img_dapi_mask = erase_solitary(img_dapi_mask)
                    if not os.path.exists(path_output_segmentaton +"erase_solitary/"):
                        os.mkdir(path_output_segmentaton +"erase_solitary/")"""
                    #tifffile.imwrite(path_output_segmentaton +"erase_solitary/"+  "dapi_maskdapi_"+ f , data=img_dapi_mask , dtype=img_dapi_mask .dtype)
    
    
                img = tifffile.imread(path_to_dapi +"dapi_"+ f)
                
                spots_568 = np.load(args.path_to_czi_folder + "detected_spot_3d"+
                        "/" + "AF568_" + f[:-5] + 'array.npy')
                spots_647 = np.load(args.path_to_czi_folder + "detected_spot_3d"+
                        "/" + "AF647_" + f[:-5] + 'array.npy')
                if args.remove_overlaping:
                    new_spots_568, removed_spots_568, new_spots_647, removed_spots_647 = erase_overlapping_spot(spots_568, 
                                                                                                            spots_647, kk_568 = args.kk_568, 
                                                                                                            kk_647 = args.kk_647)
                    spots_568_old = spots_568.copy()
                    spots_647_old = spots_647.copy()
                    spots_568 = erase_point_in_cluster_2Dalphashape(new_spots_568, removed_spots_568, eps=args.epsi_alphashape_cy3, 
                                                           min_samples = 4, min_cluster_size=10, xi=0.05)
                    spots_647 = erase_point_in_cluster_2Dalphashape(new_spots_647, removed_spots_647, eps=args.epsi_alphashape_cy5, 
                                                       min_samples = 4, min_cluster_size=10, xi=0.05)
                    spots_568, spots_647 = np.array(spots_568) , np.array(spots_647)
                if img_dapi_mask.ndim == 2:
                    spots_568 = np.array([[s[1],s[2]] for s in list(spots_568)])
                    spots_647 = np.array([[s[1],s[2]] for s in list(spots_647)])
                print(args.epsi_cluster_cy3)
                print(args.epsi_cluster_cy5)
                labels_568 = computer_optics_cluster(spots_568, eps=args.epsi_cluster_cy3, min_samples = 4, min_cluster_size=4, xi=0.05)
                labels_647 = computer_optics_cluster(spots_647, eps=args.epsi_cluster_cy5, min_samples = 4, min_cluster_size=4, xi=0.05)


                if img_dapi_mask.ndim == 3:
    
                    nuclei_568_1, positive_cluster_568,  negative_cluster_568 = cluster_over_nuclei_3D_convex_hull(labels_568, 
                                                                                            spots_568, img_dapi_mask, 
                                                                                            iou_threshold = args.overlapping_cy3)
                    nuclei_647_1, positive_cluster_647,  negative_cluster_647 =  cluster_over_nuclei_3D_convex_hull(labels_647, spots_647, 
                                                                                                 img_dapi_mask, iou_threshold = args.overlapping_cy5)
                  
                        

                    
                    nb_no_rna = len(np.unique(img_dapi_mask)) - len(set(nuclei_647_1).union(set(nuclei_568_1)))
                    nb_cy3 = len(set(nuclei_568_1)-set(nuclei_647_1))
                    nb_cy5 = len(set(nuclei_647_1)-set(nuclei_568_1))
                    nb_both = len(set(nuclei_647_1).intersection(set(nuclei_568_1)))
                    
                    dico_stat[f] = [len(np.unique(img_dapi_mask)), nb_no_rna, nb_cy3, nb_cy5, nb_both, 
                                    positive_cluster_568, positive_cluster_647, negative_cluster_568, negative_cluster_647]
                    
                    np.save(args.path_to_project + "dico_stat_2106", dico_stat)
                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                fig.suptitle(f+ "green (no rna) %s Cy3 orange %s Cy5 blue %s  uncertain purplue %s" % (str(nb_no_rna), 
                                                                                   str(nb_cy3),
                                                                                 str(nb_cy5),
                                                                                 str(nb_both)), fontsize = 20)
                    
                m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list(np.amax(img,0),
                                                        np.amax(img_dapi_mask,0), nuclei_568_1, nuclei_647_1)  
                ax.imshow(m)
                plt.show()

                
                print(f)
                print("number of green (no rna) %s" % nb_no_rna)
                print("number of yellow Cy3 %s" % nb_cy3)
                print("number of blue Cy5 %s" % nb_cy5)
                print("number of purple uncertain %s" % nb_both)
                #()
                fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" +f[:-5] +"/"
                            + "celltype_state" )
                
                if not args.remove_overlaping:
                    fig, ax = plt.subplots(1,1,  figsize=(30,20))
        
                    if img.ndim == 3:
                        ax.imshow(np.amax(img,0), cmap='gray')
                    else:
                        ax.imshow(img, cmap='gray')
                        
                    
                    for s in spots_568:
                        ax.scatter(s[-1], s[-2], c = 'red', s = 10)
            
                    
                    for s in spots_647:
                        ax.scatter(s[-1], s[-2], c = 'green', s =10)
                    #plt.show()
                    fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"
                                + "rnaspot_on_dapi_no_suppression_artifacts")
                if len(spots_568_old) > 15000 or len(spots_647_old) > 15000:
                    for path_to_rna in [[path_to_af568 + 'AF568_', spots_568, "red"] , [path_to_af647 + 'AF647_', spots_647, "green"]]:
                        fig, ax = plt.subplots(2,1,  figsize=(35,60))
                        rna = tifffile.imread(path_to_rna[0] + f)
                        ax[0].imshow(np.amax(rna,0))
                        ax[1].imshow(np.amax(rna,0))
                        #plt.show()
                    continue
                    
                if args.remove_overlaping:
                    fig, ax = plt.subplots(1,1,  figsize=(30,20))
        
                    if img.ndim == 3:
                        ax.imshow(np.amax(img,0), cmap='gray')
                    else:
                        ax.imshow(img, cmap='gray')
                        
                    
                    for s in spots_568_old:
                        ax.scatter(s[-1], s[-2], c = 'red', s = 10)
            
                    
                    for s in spots_647_old:
                        ax.scatter(s[-1], s[-2], c = 'green', s =10)
                    #plt.show()
                    fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"
                                + "rnaspot_on_dapi_no_suppression_artifacts")
                    
                    fig, ax = plt.subplots(1,1,  figsize=(30,20))
        
                    if img.ndim == 3:
                        ax.imshow(np.amax(img,0), cmap='gray')
                    else:
                        ax.imshow(img, cmap='gray')
                        
    
                        
                        
                    
                    for s in new_spots_568:
                        ax.scatter(s[-1], s[-2], c = 'red', s = 10)
            
                    
                    for s in new_spots_647:
                        ax.scatter(s[-1], s[-2], c = 'green', s =10)
                    #plt.show()
                    fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"
                                + "rnaspot_on_dapi_with_suppression_artifacts")
                
                
        
                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                plt.title(f+ " Cy3 red and cy5 green" , fontsize = 20)
                if img.ndim == 3:
                    ax.imshow(np.amax(img,0), cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')
                    
                set_cluster_568 = [el[0] for el in positive_cluster_568]
                for s_index in range(len(spots_568)):
                    if labels_568[s_index] in set_cluster_568:
                        s = spots_568[s_index]
                        ax.scatter(s[-1], s[-2], c = 'red', s = 10)
        
                set_cluster_647 = [el[0] for el in positive_cluster_647]
                for s_index in range(len(spots_647)):
                    if labels_647[s_index] in set_cluster_647:
                        s = spots_647[s_index]
                        ax.scatter(s[-1], s[-2], c = 'green', s =10)
                #plt.show()
    
                fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"
                            + "rnaspot_final_on_dapi")
                
      
        
        
                
                ###plot smfish signal
                for path_to_rna in [[path_to_af568 + 'AF568_', spots_568, "red"] , [path_to_af647 + 'AF647_', spots_647, "green"]]:
                    fig, ax = plt.subplots(2,1,  figsize=(35,60))
                    rna = tifffile.imread(path_to_rna[0] + f)
                    ax[0].imshow(np.amax(rna,0))
                    ax[1].imshow(np.amax(rna,0))
                    
    
                    for s in path_to_rna[1]:
                        ax[1].scatter(s[-1], s[-2], c = path_to_rna[2], s = 10)
                    #plt.show()
                    fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"
                            + "rnaspot_smfish" + path_to_rna[0][-6:-2])
                 ###### save clustering plot
                if img.ndim == 2:
                    nuclei = mask_image_to_rgb(img, img_dapi_mask, colors = None)
                else :
                    nuclei = mask_image_to_rgb(np.amax(img,0), np.amax(img_dapi_mask,0), colors = None)
                
                dico_plot_cluster = {"af568": [spots_568, labels_568], "af647" : [spots_647, labels_647]}
                
                for key in dico_plot_cluster.keys():
                    fig, ax = plt.subplots(1,1,  figsize=(30,20))
                    ax.imshow(nuclei)
    
                    
                    spots = dico_plot_cluster[key][0]
                    labels_200 = dico_plot_cluster[key][1]
                    for cluster in range(np.max(labels_200)):
                        color = np.random.rand(1,3)
                        for index in range(len(spots)):
                            if labels_200[index] == cluster:
                                ax.scatter(spots[index][-1], spots[index][-2], c = color, s = 18)
                                
                    for index in range(len(spots)):
                        if labels_200[index] == -1:
                            ax.scatter(spots[index][-1], spots[index][-2], c = 'red', s = 12)
                    fig.suptitle(f+ " cluster rna" + str(key) + ", red dot are outlier", fontsize = 20)
                    #plt.show()
                    fig.savefig(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"
                                + "rna_cluster"+ str(key))
                    
                    array_to_save_cell = np.array([nuclei_568_1, positive_cluster_568,  negative_cluster_568,
                                                   nuclei_647_1, positive_cluster_647,  negative_cluster_647])
                    
                    np.save(args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"  +'result', array_to_save_cell)
                    args.epsi_cluster_cy3 = "e"
                    args.overlapping_cy3 = "e"
                    args.epsi_cluster_cy5 = "e"
                    args.overlapping_cy5 = "e"
                if nb_fileee % 10 ==0:
                    plt.show(block=False)
                    plt.clf()
                    plt.cla()
                    plt.close("all")
                    plt.clf()
                    plt.cla()
                del fig
            
                np.save(args.path_to_project + "dico_stat_2106", dico_stat)
            #dico_stat_to_exel(dico_stat, path_name = args.path_to_project + "summary_1305")
            #with open(args.path_to_project + 'commandline_args.txt', 'w') as lmp:
             #   json.dump(args.__dict__, lmp, indent=2)"""
#%%


#%%
list_folder = [
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/", #okd

"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",#okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/"
]

list_folder = [#"210426_repeat3/", #ok
"200828-NIvsIR5M/00_Capillary_EC/", #okd
"200828-NIvsIR5M/00_Large_Vessels/", #okd
"200828-NIvsIR5M/00_Macrophages/", #okd
"200908_CEC/", #okd

"201030_fridyay/", #ok
"210205_Prolicence/aCap_prolif/", #okd
"210205_Prolicence/aCap_senes/", #okd
"210219_myo_fibros_y_macrophages/",#okd
"210412_repeat_fibro/IR5M/", #okd
"210412_repeat_fibro/NI/", #okd
"210413_rep2/", #okd
"210425_angiogenesis/", #ok
"200908_fibrosis/", #ok
"201127_AM_fibro/", #okd
#"210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/"
]



l_param = []

for l in list_folder:
    l_param.append([l])
for l in l_param[5:]:
    main(l)
    
    
number_processes = 6
pool = multiprocessing.Pool(number_processes)
results = pool.map_async(main, l_param)
pool.close()
pool.join()


#%%
list_folder = [
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/", #okd

"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",#okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/", #ok
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/", #okd
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/"
]
for folder in list_folder:
    print(len(np.load(list_folder,'dico_stat_2106.npy', allow_pickle = True).item()))
    onlyfiles = [f for f in listdir(folder+ "tiff_data/" + "af647/") if isfile(join(folder+ "tiff_data/" + "af647/", f)) and f[-1] == "f" ]

#%%
"""

    from tqdm import tqdm
    def save_mask(l_params):
        t = time.time()
        print(l_params)
        folder_name, f = l_params[0], l_params[1]
        local_path = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/"
        path_output_segmentaton = local_path + folder_name + "tiff_data/" + "predicted_mask_dapi/"
    
        disk_path = local_path
        path_to_watersheld = disk_path + folder_name + "tiff_data/watersheld/"
        if os.path.exists(path_to_watersheld +"watersheld"+f+".npy"):
            print("continue")
            return

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
    list_folder.reverse()
    for folder_name in list_folder:
        print(folder_name)
        local_path = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/"
        path_output_segmentaton = local_path + folder_name + "tiff_data/" + "predicted_mask_dapi/"
        
        disk_path = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/"
        path_to_watersheld = disk_path + folder_name + "tiff_data/watersheld/"
        if not os.path.exists(path_to_watersheld):
             os.mkdir(path_to_watersheld )
        onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
        l_params = []
        for f in onlyfiles:
            l_params.append([folder_name, f])
        number_processes = 4
        pool = multiprocessing.Pool(number_processes)
        results = pool.map_async(save_mask, l_params)
        pool.close()
        pool.join()
    #%%

    
#save mask
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
  
for folder_name in list_folder:
      print(folder_name)
      local_path = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/"
      path_output_segmentaton = local_path + folder_name + "tiff_data/" + "predicted_mask_dapi/"
      
      disk_path = "/media/tom/Elements/to_take_2/"
      path_to_watersheld = "/media/tom/Elements/to_take_2/" + folder_name + "tiff_data/erase_solitary/"
      if not os.path.exists(path_to_watersheld):
           os.mkdir(path_to_watersheld )
      onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
      for f in onlyfiles:
          if os.path.exists(path_to_watersheld +f):
              print("ok")
              continue
          print(f)
          img_dapi_mask = tifffile.imread(path_output_segmentaton  + f)
          img_dapi_mask = erase_solitary(img_dapi_mask)
          tifffile.imwrite(path_to_watersheld +f, img_dapi_mask)
#%%
####
img = tifffile.imread(path_to_dapi +"dapi_"+ f)
d = np.load("/home/tom/Bureau/annotation/cell_type_annotation/all_anotation/01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_01.npy",
        allow_pickle=True).item()
nuclei_568_1 = d["af568"]
nuclei_647_1 = d["af647"]
img_dapi_mask = d["mask"]
m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list(np.amax(img,0),
                                                np.amax(img_dapi_mask,0), nuclei_568_1 , nuclei_647_1, colors)       

fig, ax = plt.subplots(1,1,  figsize=(30,35))

ax.imshow(m)
#%%
dico_plot_cluster = {"af568": [spots_568, labels_568], "af647" : [spots_647, labels_647]}

for key in list(dico_plot_cluster.keys())[:1]:
    spots = dico_plot_cluster[key][0]
    labels_200 = dico_plot_cluster[key][1]
    for cluster in range(np.max(labels_200)+1):
        color = np.random.rand(1,3)
        for index in range(len(spots)):
            if labels_200[index] == cluster:
                ax.scatter(spots[index][-1], spots[index][-2], c = color, s = 15)
#%%
m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list(np.amax(img,0),
                                                np.amax(img_dapi_mask,0), [] , nuclei_647_1, colors)       

fig, ax = plt.subplots(1,1,  figsize=(30,35))

ax.imshow(m)
dico_plot_cluster = {"af568": [spots_568, labels_568], "af647" : [spots_647, labels_647]}

for key in list(dico_plot_cluster.keys())[1:]:

    
    spots = dico_plot_cluster[key][0]
    labels_200 = dico_plot_cluster[key][1]
    for cluster in range(np.max(labels_200)+1):
        color = np.random.rand(1,3)
        for index in range(len(spots)):
            if labels_200[index] == cluster:
                ax.scatter(spots[index][-1], spots[index][-2], c = color, s = 15)


#%%
dico_annotat = {}
dico_annotat = {"mask" : img_dapi_mask}
#%%

for nuc_index  in range(len(nuclei_647_1)):#np.unique(img_dapi_mask):
    nuc = nuclei_647_1[nuc_index]
    overlap = positive_cluster_647[nuc_index][1]
    print((nuc, positive_cluster_647[nuc_index][-1]))
    fig, ax = plt.subplots(1,1,  figsize=(10,11))
    fig.suptitle(str(nuc) +"  " + str(overlap), fontsize = 20)
    ax.imshow(np.amax(img,0), alpha = 0.5, cmap = 'gray')
    ax.imshow(np.amax(img_dapi_mask ==  nuc,0), alpha = 0.5,  cmap = 'gray')
    plt.show()

#%%
dico_annotat["af647"]   = [8.0,9.0,20.0,60.0,49.0,74.0,86.0,91.0,106.0,11.0,37.0,13.0,22.0,26.0,33.0,23.0]
#%%
overlap = 0
for nuc_index in range(len(nuclei_568_1)):
        nuc = nuclei_568_1[nuc_index]
        overlap = positive_cluster_568[nuc_index][1]
        fig, ax = plt.subplots(1,1,  figsize=(10,11))
        fig.suptitle(str(nuc) +"  " + str(overlap), fontsize = 20)
        ax.imshow(np.amax(img,0), alpha = 0.5, cmap = 'gray')
        ax.imshow(np.amax(img_dapi_mask ==  nuc,0), alpha = 0.5,  cmap = 'gray')
        plt.show()

#%%

#[63.0,98.0,109.0,76.0,94.0,18.0,33.0,36.0,80.0,99.0,110.0,13.0,26.0,22.0,111.0,130.0]    
    
nuclei_568_1_true = [26.0,
 8.0,
 9.0,
 20.0,
 60.0,
 23.0,
 49.0,
 74.0,
 86.0,
 91.0,
 106.0,
 11.0,
 33.0]

dico_annotat["af568"]  = nuclei_568_1_true

#%%
for nuc in np.unique(img_dapi_mask):
    if nuc in nuclei_647_1:
        fig, ax = plt.subplots(1,1,  figsize=(10,11))
        fig.suptitle(str(nuc), fontsize = 20)
        ax.imshow(np.amax(img,0), alpha = 0.5, cmap = 'gray')
        ax.imshow(np.amax(img_dapi_mask ==  nuc,0), alpha = 0.5,  cmap = 'gray')
        plt.show()


#%%


#%%
print(dico_annotat)
np.save("/home/tom/Bureau/annotation/cell_type_annotation/from_rna_anno/" 
        +"/03_IR5M2201()_Pecam1-Cy5_Apln-Cy3_02", dico_annotat)


#Information|Image|Channel|Name #1 = Cy5
#Information|Image|Channel|Name #2 = Cy3
#%%

d = np.load("/home/tom/Bureau/annotation/cell_type_annotation/3105/08_IR5M_Fibin-Cy3_Serpine1-Cy5_14.npy",
        allow_pickle=True).item()
"""

