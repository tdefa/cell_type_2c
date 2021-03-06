# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Mar 31 11:27:43 2021

@author: thomas
"""
#%%

import time
import argparse
import os
from os import listdir

from os.path import isfile, join
from matplotlib import pyplot as plt
import tifffile
from utils.czi_to_tiff import preprare_tiff

from cellpose import models

from run_seg import segment_nuclei

from spots.spot_detection import computer_optics_cluster
from spots.spot_detection import cluster_over_nuclei_3D_convex_hull, spot_detection_for_clustering
from spots.spot_detection import mask_image_to_rgb2D_from_list_green_cy3_red_cy5_both_blue_grey
from spots.spot_detection import mask_image_to_rgb2D_from_list_orange_cy3_other_grey, mask_image_to_rgb2D_from_list_orange_cy5_other_grey
from spots.plot import mask_image_to_rgb

from spots.post_processing import erase_solitary

from spots.erase_overlapping_spot import erase_point_in_cluster_2Dalphashape, erase_overlapping_spot

#from compute_spatial_state import compute_average_size
import numpy as np
import warnings

import gc

warnings.filterwarnings("ignore")

#%%    main



def main(list_folder, args):
    

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




    for folder_index in range(len(list_folder)):
        print(list_folder[folder_index])
        print(list_folder)
        folder = list_folder[folder_index]
        path_to_save_fig = args.path_to_project  + "figure/"
        path_to_czi_folder_c = args.path_to_czi_folder + list_folder[folder_index]
        path_to_project_c = args.path_to_project  + list_folder[folder_index]

            
        if not os.path.exists(path_to_czi_folder_c + "tiff_data/"):
            os.mkdir(path_to_czi_folder_c + "tiff_data/")
        if not os.path.exists(path_to_czi_folder_c + "tiff_data/" + "dapi/"):
             os.mkdir(path_to_czi_folder_c+ "tiff_data/" + "dapi/")
        if not os.path.exists(path_to_czi_folder_c+ "tiff_data/" + "af568/"):
             os.mkdir(path_to_czi_folder_c+ "tiff_data/" + "af568/")
        if not os.path.exists(path_to_czi_folder_c + "tiff_data/" + "af647/"):
             os.mkdir(path_to_czi_folder_c + "tiff_data/" + "af647/")
        
        path_to_czi = path_to_czi_folder_c
        path_to_dapi = path_to_czi_folder_c + "tiff_data/" + "dapi/"
        path_to_af647 = path_to_czi_folder_c + "tiff_data/" + "af647/"
        path_to_af568 = path_to_czi_folder_c + "tiff_data/" + "af568/"
        path_output_segmentaton = path_to_czi_folder_c + "tiff_data/" + "predicted_mask_dapi/"

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
        ################
        # Nuclei segmentation
        ################
        if args.segmentation:
            model = models.Cellpose(gpu=args.gpu, model_type='nuclei', net_avg = False)
            # ##parameter
            dico_param = {}
            dico_param["diameter"]  = args.diameter
            dico_param["flow_threshold"] = args.flow_threshold
            dico_param["do_3D"] = args.do_3D
            dico_param["mip"] = args.mip
            dico_param["projected_focused"] = False 
            dico_param["stitch_threshold"] = args.stitch_threshold
            segment_nuclei(path_to_dapi  , path_output_segmentaton, dico_param, model)   
        ###
        # check that allthe nuclei are segmented
        ###
        onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
        onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

        assert len(onlyfiles) <= len([f for f in listdir(path_to_dapi ) if isfile(join(path_to_dapi, f)) and f[-1] == "f" ])

        if args.spot_detection:
             print("spotdetection")
             if not os.path.exists(path_to_project_c  + "detected_spot_3d"+"/"):
                  os.mkdir(path_to_project_c  + "detected_spot_3d"+"/")
             dico_threshold = spot_detection_for_clustering(
                                        sigma = (1.25, 1.25, 1.25),
                                        float_out= False,
                                        rna_path = [path_to_af568+'AF568_'],
                                        path_output_segmentaton = path_output_segmentaton,
                                        threshold_input = dico_cy3,
                                        output_file = path_to_project_c  + "detected_spot_3d"+"/",)
             np.save(path_to_project_c  + 'dico_threshold_AF568.npy', dico_threshold)
             print(dico_threshold)
             dico_threshold = spot_detection_for_clustering(sigma = (1.35, 1.35, 1.35), float_out= True,
                                           rna_path = [path_to_af647+'AF647_'],
                                           path_output_segmentaton = path_output_segmentaton,
                                      threshold_input = dico_cy5,
                                      output_file = path_to_project_c  + "detected_spot_3d"+"/",)
             np.save(path_to_project_c  +'dico_threshold_AF647.npy', dico_threshold)
             print(dico_threshold)
    
            
        
        if args.classify:
            print("classify")
            if not os.path.exists(path_to_project_c  + "plot_clustering_artifact_sup/"):
                 os.mkdir(path_to_project_c  + "plot_clustering_artifact_sup/")
            try :
                dico_stat =  np.load(path_to_project_c  + args.dico_name_save +'.npy', allow_pickle = True).item()
            except Exception as e:
                print(e)
                dico_stat = {}
            if not os.path.exists( path_to_project_c  + "plot_clustering_artifact_sup/"):
                 os.mkdir(path_to_project_c  + "plot_clustering_artifact_sup/")
            try :
                dico_label_cluster =  np.load(path_to_project_c  + "dico_label_cluster" +'.npy', allow_pickle = True).item()
            except Exception as e:
                print(e)
                dico_label_cluster = {}
            nb_fileee = 0
            t = time.time()
            print(onlyfiles)
            for f in onlyfiles[:]:
            #    if f == 'IR1M_aCapCy3_Mki67Cy5_03.tiff':
             #       continue
                print(list_folder[folder_index])
                print(nb_fileee)
                print(time.time() -t)
                print(f[:-5])
                if os.path.isfile(path_to_save_fig + folder + "segmentation/" +f[:-5] + ".png"):
                    print(f + "exist already")
                    continue
    
                t = time.time()
                nb_fileee += 1

                args.epsi_cluster_cy3 = 'probe not reconize yet'
                args.epsi_cluster_cy5 = 'probe not reconize yet'

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
                # load mask, remove solitary and load spot detection
                ####
                if not os.path.exists( path_to_project_c  + "plot_clustering_artifact_sup/" + f[:-5] +"/"):
                    os.mkdir(path_to_project_c  + "plot_clustering_artifact_sup/"+ f[:-5] +"/")
                print(f)
                img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f)
                img_dapi_mask = erase_solitary(img_dapi_mask)
                img = tifffile.imread(path_to_dapi +"dapi_"+ f)
                
                spots_568 = np.load(path_to_czi_folder_c + "detected_spot_3d"+
                        "/" + "AF568_" + f[:-5] + 'array.npy')
                spots_647 = np.load(path_to_czi_folder_c + "detected_spot_3d"+
                        "/" + "AF647_" + f[:-5] + 'array.npy')

                ######
                # erase ovevalaping spot and spot in alphashape of artefact
                ######
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
                ########
                # compute clustered dbscan
                #########
                if img_dapi_mask.ndim == 2:
                    spots_568 = np.array([[s[1],s[2]] for s in list(spots_568)])
                    spots_647 = np.array([[s[1],s[2]] for s in list(spots_647)])


                print(args.epsi_cluster_cy3)
                print(args.epsi_cluster_cy5)
                print(len(spots_568))
                print(len(spots_647))
                labels_568 =  np.array([-1] * len(spots_568))
                labels_647 =  np.array([-1] * len(spots_647))
                if type(args.epsi_cluster_cy3) == int:
                    labels_568 = computer_optics_cluster(spots_568, eps=args.epsi_cluster_cy3, min_samples=4,
                                                         min_cluster_size=4, xi=0.05)
                if type(args.epsi_cluster_cy5) == int:
                    labels_647 = computer_optics_cluster(spots_647, eps=args.epsi_cluster_cy5, min_samples=4,
                                                         min_cluster_size=4, xi=0.05)

                ##########
                # classify cell type
                #########
                if img_dapi_mask.ndim == 3:
    
                    nuclei_568_1, positive_cluster_568,  negative_cluster_568 = cluster_over_nuclei_3D_convex_hull(labels_568, 
                                                                                            spots_568,
                                                                                            img_dapi_mask,
                                                                                            iou_threshold = args.overlapping_cy3)
                    nuclei_647_1, positive_cluster_647,  negative_cluster_647 =  cluster_over_nuclei_3D_convex_hull(labels_647, spots_647, 
                                                                                                 img_dapi_mask, iou_threshold = args.overlapping_cy5)


                    nb_no_rna = len(np.unique(img_dapi_mask)) - len(set(nuclei_647_1).union(set(nuclei_568_1)))
                    nb_cy3 = len(set(nuclei_568_1)-set(nuclei_647_1))
                    nb_cy5 = len(set(nuclei_647_1)-set(nuclei_568_1))
                    nb_both = len(set(nuclei_647_1).intersection(set(nuclei_568_1)))
                    
                    dico_stat[f] = [len(np.unique(img_dapi_mask)), nb_no_rna, nb_cy3, nb_cy5, nb_both, 
                                    positive_cluster_568, positive_cluster_647, negative_cluster_568, negative_cluster_647]

                    dico_label_cluster[f] = [labels_568, labels_647]
                    np.save(path_to_project_c + "dico_label_cluster", dico_label_cluster)
                    np.save(path_to_project_c  + args.dico_name_save, dico_stat)
                    print("classification done")
                if not args.save_plot:
                    continue

                if not os.path.exists(path_to_save_fig + folder ):
                     os.mkdir(path_to_save_fig + folder )
                if not os.path.exists(path_to_save_fig):
                     os.mkdir(path_to_save_fig)
                if os.path.isfile(path_to_save_fig + folder + "convex_hull/Cy5_convexhull" + f[:-5]):
                    print(f + "exist already")
                    continue
                ####
                # plot final classification
                ######
                if not os.path.exists(path_to_save_fig + folder + "classif/"):
                    os.mkdir(path_to_save_fig + folder + "classif/")
                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                fig.suptitle(f+ " grey (no rna) %s, Cy3 green %s, Cy5 red %s,  Both blue %s" % (str(nb_no_rna),
                                                                               str(nb_cy3),
                                                                             str(nb_cy5),
                                                                             str(nb_both)), fontsize = 20)

                m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list_green_cy3_red_cy5_both_blue_grey(np.amax(img,0),
                                                    np.amax(img_dapi_mask,0), nuclei_568_1, nuclei_647_1)
                ax.imshow(m)
                fig.savefig(path_to_save_fig + folder + "classif/green_cy3_red_cy5" +f[:-5])
                print("classif save")


                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                fig.suptitle(f+ "Cy3 orange %s, other grey %s " % ( str(nb_cy3),
                                                                             str(nb_cy5+nb_no_rna+nb_both)), fontsize = 20)

                m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list_orange_cy3_other_grey(np.amax(img,0),
                                                    np.amax(img_dapi_mask,0), nuclei_568_1, nuclei_647_1)
                ax.imshow(m)
                fig.savefig(path_to_save_fig + folder + "classif/orange_cy3" +f[:-5])

                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                fig.suptitle(f+ "Cy5 orange %s, other grey %s " % ( str(nb_cy5),
                                                                             str(nb_cy3+nb_no_rna+nb_both)), fontsize = 20)

                m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list_orange_cy5_other_grey(np.amax(img,0),
                                                    np.amax(img_dapi_mask,0), nuclei_568_1, nuclei_647_1)
                ax.imshow(m)
                fig.savefig(path_to_save_fig + folder + "classif/orange_cy5" +f[:-5])



                #####
                # plot segmentation
                #####
                if not os.path.exists(path_to_save_fig + folder + "segmentation/"):
                    os.mkdir(path_to_save_fig + folder + "segmentation/")

                fig, ax = plt.subplots(1,1,  figsize=(30,20))

                m =  mask_image_to_rgb(np.amax(img, 0), np.amax(img_dapi_mask, 0))
                ax.imshow(m)
                fig.savefig(path_to_save_fig + folder + "segmentation/" +f[:-5])

                print(len(spots_568))
                print(len(spots_647))
                print("segmentation save")
                continue
                #####
                # plot dapi_slpot ### Cy3 green cy5 orange
                #####

                if not os.path.exists(path_to_save_fig + folder + "dapi_spots/"):
                    os.mkdir(path_to_save_fig + folder + "dapi_spots/")


                #### raw spot detection
                path_to_af568 = path_to_czi_folder_c + "tiff_data/" + "af568/"

                path_to_af647 = path_to_czi_folder_c + "tiff_data/" + "af647/"
                cy3_im = tifffile.imread(path_to_af568 + "AF568_" +f)
                cy5_im = tifffile.imread(path_to_af647 + "AF647_" +f)


                fig, ax = plt.subplots(1, 1, figsize=(30, 20))
                plt.title(f + " Cy3 fish green spots" , fontsize=20)
                if img.ndim == 3:
                    ax.imshow(np.amax(cy3_im, 0))
                else:
                    ax.imshow(cy3_im)
                for s in spots_568:
                    ax.scatter(s[-1], s[-2], c='green', s=10)
                fig.savefig(path_to_save_fig+ folder + "dapi_spots/smfish_cy3" + f[:-5] )

                fig, ax = plt.subplots(1, 1, figsize=(30, 20))
                plt.title(f + " Cy5 fish green spots" , fontsize=20)
                if img.ndim == 3:
                    ax.imshow(np.amax(cy5_im, 0))
                else:
                    ax.imshow(cy5_im)
                for s in spots_647:
                    ax.scatter(s[-1], s[-2], c='green', s=10)
                fig.savefig(path_to_save_fig+ folder + "dapi_spots/smfish_cy5" + f[:-5])
                print("spot save")

                fig, ax = plt.subplots(1, 1,  figsize=(30,20))
                plt.title(f+ " Cy3 green and cy5 red" , fontsize = 20)
                if img.ndim == 3:
                    ax.imshow(np.amax(img,0), cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')

                set_cluster_568 = [el[0] for el in positive_cluster_568]
                for s_index in range(len(spots_568)):
                    if labels_568[s_index] in set_cluster_568:
                        s = spots_568[s_index]
                        ax.scatter(s[-1], s[-2], c = 'green', s = 10)

                set_cluster_647 = [el[0] for el in positive_cluster_647]
                for s_index in range(len(spots_647)):
                    if labels_647[s_index] in set_cluster_647:
                        s = spots_647[s_index]
                        ax.scatter(s[-1], s[-2], c = 'red', s =10)

                fig.savefig(path_to_save_fig+ folder + "dapi_spots/green_red" + f[:-5])
                print("spots green save")

                ### Cy3 orange
                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                plt.title(f+ " Cy3 channel, orange" , fontsize = 20)
                if img.ndim == 3:
                    ax.imshow(np.amax(img,0), cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')

                set_cluster_568 = [el[0] for el in positive_cluster_568]
                for s_index in range(len(spots_568)):
                    if labels_568[s_index] in set_cluster_568:
                        s = spots_568[s_index]
                        ax.scatter(s[-1], s[-2], c = 'orange', s = 10)

                fig.savefig(path_to_save_fig+ folder + "dapi_spots/orange_cy3" + f[:-5])
                print("spots orange save")


                ###cy5 orange
                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                plt.title(f+ " cy5 channel, orange" , fontsize = 20)
                if img.ndim == 3:
                    ax.imshow(np.amax(img,0), cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')

                set_cluster_647 = [el[0] for el in positive_cluster_647]
                for s_index in range(len(spots_647)):
                    if labels_647[s_index] in set_cluster_647:
                        s = spots_647[s_index]
                        ax.scatter(s[-1], s[-2], c = 'orange', s = 10)
                fig.savefig(path_to_save_fig+ folder + "dapi_spots/orange_cy5" + f[:-5])
                print("spots orange save")

                #### point cloud
                if not os.path.exists(path_to_save_fig + folder + "convex_hull/"):
                    os.mkdir(path_to_save_fig + folder + "convex_hull/")
                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                plt.title(f+ " Cy3 channel" , fontsize = 20)
                if img.ndim == 3:
                    ax.imshow(np.amax(img,0), cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')

                set_cluster_568 = [el[0] for el in positive_cluster_568] + [el[0] for el in negative_cluster_568]
                for s_index in range(len(spots_568)):
                    if labels_568[s_index] in set_cluster_568:
                        s = spots_568[s_index]
                        ax.scatter(s[-1], s[-2], c = 'orange', s = 10)

                from scipy.spatial import ConvexHull
                for c in set_cluster_568:
                    point_cloud = []
                    for s_index in range(len(spots_568)):
                        if labels_568[s_index] == c :
                            point_cloud.append([spots_568[s_index][2], spots_568[s_index][1]])
                    points = np.array(point_cloud)
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'c')
                        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
                fig.savefig(path_to_save_fig + folder + "convex_hull/Cy3_convexhull" + f[:-5] )


                fig, ax = plt.subplots(1,1,  figsize=(30,20))
                plt.title(f+ " Cy5 channel" , fontsize = 20)
                if img.ndim == 3:
                    ax.imshow(np.amax(img,0), cmap='gray')
                else:
                    ax.imshow(img, cmap='gray')

                set_cluster_647 = [el[0] for el in positive_cluster_647] + [el[0] for el in positive_cluster_647]
                for s_index in range(len(spots_647)):
                    if labels_647[s_index] in set_cluster_647:
                        s = spots_647[s_index]
                        ax.scatter(s[-1], s[-2], c = 'orange', s = 10)

                from scipy.spatial import ConvexHull
                for c in set_cluster_647:
                    point_cloud = []
                    for s_index in range(len(spots_647)):
                        if labels_647[s_index] == c :
                            point_cloud.append([spots_647[s_index][2], spots_647[s_index][1]])
                    points = np.array(point_cloud)
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], 'c')
                        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
                fig.savefig(path_to_save_fig + folder + "convex_hull/Cy5_convexhull" + f[:-5] )
                print("convex hull save")
                plt.show()
                plt.close("all")
                gc.collect()
                del fig
                del ax
                gc.collect()

#%%

if __name__ == '__main__':
    list_folder = [
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/", #ok
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/", #ok
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",#okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/", #okd
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/", #ok
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/", #ok
        # "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3"
        #  "01_lamp3-cy3_normal_01-czi_2021-10-25_1543/",
        # "02_lamp3-cy3_normal-trueview_02-czi_2021-10-25_1544/",

        "gCap prolif/",
        "gCap senes/",
        "aCap prolif/",
        "aCap senes/",

    ]
    print("in the main")
    import torch
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-ptz', "--path_to_czi_folder",
                        type=str,
                        default="/home/tom/Bureau/210205_Prolicence/",
                        help='path_to_czi folder')

    parser.add_argument('-ptp', "--path_to_project", type=str,
                        default="/home/tom/Bureau/210205_Prolicence/",
                        help='path_to_project')

    parser.add_argument('-dns', "--dico_name_save", type=str,
                        default="dico_stat_2810",
                        help='path_to_project')
    parser.add_argument("--list_folder", nargs="+", default= list_folder)

    ###cellpose arg
    parser.add_argument('-d', "--diameter", type=float, default=None, help='')
    parser.add_argument('-ft', "--flow_threshold", type=float, default=0.55, help='')
    parser.add_argument('-d3', "--do_3D", type=bool, default=False, help='')
    parser.add_argument('-m', "--mip", type=bool, default=False, help='')
    parser.add_argument('-st', "--stitch_threshold", type=float, default=0.4, help='')
    parser.add_argument('-er', "--erase_solitary", type=int, default=1, help='')

    parser.add_argument('-prczi', "--prepare_czi", type=int, default=0, help='')
    parser.add_argument('-sg', "--segmentation", type=int, default=0, help='')
    parser.add_argument("--spot_detection", type=int, default=0, help='')
    parser.add_argument("--classify", type=int, default=1, help='')
    parser.add_argument("--save_plot", type=int, default=1, help='')

    # not used parser.add_argument("--clustering", type=int, default=0, help='')

    parser.add_argument("--epsi_cluster_cy3", default="??", help='')
    parser.add_argument("--epsi_cluster_cy5", default="e", help='')

    parser.add_argument("--epsi_alphashape_cy3", type=int, default=25, help='')
    parser.add_argument("--epsi_alphashape_cy5", type=int, default=25, help='')

    parser.add_argument("--overlapping_cy3", default="e", help='')
    parser.add_argument("--overlapping_cy5", default="e", help='')

    parser.add_argument("--remove_overlaping", type=int, default=1, help='')
    parser.add_argument("--gpu", type=int, default=0, help='')

    parser.add_argument("--kk_568", type=int, default=3)
    parser.add_argument("--kk_647", type=int, default=3)
    parser.add_argument("--port", default=39949)
    parser.add_argument("--mode", default='client')
    args = parser.parse_args()
    print(args)
    main(args.list_folder, args)

