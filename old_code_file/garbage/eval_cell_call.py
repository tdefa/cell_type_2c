

import pandas as pd
import argparse
import os
from os import listdir
from os.path import isfile, join
import czifile as zis
from matplotlib import pyplot as plt
import tifffile
import numpy as np

from run_seg import segment_nuclei

from spot_detection import mask_image_to_rgb_bis2d, mask_image_to_rgb_bis3d, computer_optics_cluster
from spot_detection import cluster_over_nuclei_3D_convex_hull, spot_detection_for_clustering, cluster_over_nuclei, mask_image_to_rgb2D_from_list
from spot_detection import cluster_in_nuclei, cluster_over_nuclei_3D

from post_processing import erase_solitary

from spots.erase_overlapping_spot import erase_point_in_cluster_2Dalphashape, erase_overlapping_spot


import numpy as np
import warnings

import random

#%%

def compute_average_size(l_d):
    """l_d: is the list of tuple (cluster number, overlapp, cluster volume)
    for a list of 
    """
    dico_int = {}
    for tup in l_d:
        dico_int[tup[0]] = []
    for tup in l_d:
        dico_int[tup[0]].append(tup[2])
    total_sum = 0
    for k in dico_int.keys():
        total_sum += dico_int[k][0] #sum only once cloud that are in many nuclei
    if len(l_d) > 0:
        return (total_sum / len(l_d)) * 103 * (10**(-9))
    return "not defined"
     

def dico_stat_to_exel(dico_stat, path_name):
    dico = {}
    dico["name"] = dico_stat.keys()
    dico['total nb Nuclei']  = [dico_stat[k][0] for k in dico_stat.keys()]
    dico['green (no rna)'] = [dico_stat[k][1] for k in dico_stat.keys()]
    dico['yellow Cy3'] = [dico_stat[k][2] for k in dico_stat.keys()]
    dico['blue Cy5'] = [dico_stat[k][3] for k in dico_stat.keys()]
    dico['purple uncertain'] = [dico_stat[k][4] for k in dico_stat.keys()]
    dico["average volume point cloud Cy3 (µm3) "] = [compute_average_size(dico_stat[k][5]) for k in dico_stat.keys()]
    dico["average volume point cloud Cy5 (µm3)"] = [compute_average_size(dico_stat[k][6]) for k in dico_stat.keys()]
    df = pd.DataFrame.from_dict(dico)
    df.to_excel(path_name + '.xls')



dico_cy3 = {"02_NI1230_Lamp3-Cy5_Pdgfra-Cy3_04.tiff" : 45}
dico_cy5 = {"02_NI1230_Lamp3-Cy5_Pdgfra-Cy3_08.tiff" : 8,
             "01_IR5M1236_Lamp3-Cy5_Pdgfra-Cy5_04.tiff" : 7,
             "05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_08.tiff" : 9,
             "05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_01.tiff" : 9,}


list_folder = [
 "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/", #ok spot ok image, on image is wrong
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/", #pb to rerun
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/", #ok spot
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/", 
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/", ##pb
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/",
"/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/",

]
dico_param_probes = {"Lamp3": (32, 0.42),
              "Pdgfra" : (35, 0.42),
              "Chil3": (15, 0.55),
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
              "CEC": (0, 2),
              }

onlyfiles_ano2 = [f for f in listdir("/home/tom/Bureau/annotation/cell_type_annotation/all_anotation")  ]
onlyfiles_ano2 = [onlyfiles_ano2[i][:-4] for i in range(len(onlyfiles_ano2))]

['01_NI2323()_Pecam1-Cy5_Apln-Cy3_03',
 '02_IR5M_C3ar1-Cy3_Chil3-Cy5_009',
 '02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_009',
 '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_01',
 '04_IR5M_Hhip-Cy3_Pdgfra-Cy5_003',
 '02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_010',
 '04_IR5M_Hhip-Cy3_Pdgfra-Cy5_014',
 '02_IR5M_Chil3-Cy3_Mki67-Cy5_05',
 '08_IR5M_Fibin-Cy3_Serpine1-Cy5_01',
 '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_06',
 '12_IR5M_Cap-Cy3_Mki67-Cy5_009',
 '05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_03',
 '05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_01',
 '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_02',
 '08_IR5M_Fibin-Cy3_nop-Cy5_19',
 '02_IR5M_C3ar1-Cy3_Chil3-Cy5_011',
 '03_IR5M2201()_Pecam1-Cy5_Apln-Cy3_05',
 '04_IR5M_Chil3-Cy3_Serpine1-Cy5_02',
 '02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_008',
 '01_NI_C3ar1-Cy3_Chil3-Cy5_006',
 '02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_007',
 '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_08',
 '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_04',
 '08_IR5M_Fibin-Cy3_Serpine1-Cy5_19',
 '08_IR5M_Fibin-Cy3_Serpine1-Cy5_14',
 '12_IR5M_Cap-Cy3_Mki67-Cy5_008',
 '03_IR5M2201()_Pecam1-Cy5_Apln-Cy3_02',
 '02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_006',
 '03_NI_Chil3-Cy3_Serpine1-Cy5_01',
 '03_NI_Hhip-Cy3_Pdgfra-Cy5_004',
 '12_IR5M_Ptprb-Cy3_Serpine1-Cy5_04',
 'pred_vs_annota',
 '09_NI_Ptprb-Cy3_Mki67-Cy5_03',
 '02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_012',
 '10_IR5M_Ptprb-Cy3_Mki67-Cy5_02',
 '14_IR5M_Cap-Cy3_Serpine1-Cy5_011',
 '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_07',
 '210425_angiogenesisdico_stat']

dico_perf_568 = {}
dico_perf_647 = {}
list_folder_project  = list_folder
list_perf_568 = []
list_perf_647 = []
for folder_index in range(len(list_folder)):
    print(list_folder[folder_index])
    
    parser = argparse.ArgumentParser(description='test')
    ### path to datset
    parser.add_argument('-ptz',"--path_to_czi_folder", type=str,
                        default=list_folder[folder_index], help='path_to_czi folder')
    
    parser.add_argument('-ptp',"--path_to_project", type=str, 
                        default= list_folder_project[folder_index],
                        help='path_to_project')
    parser.add_argument('--ptvplo' ,type=str, 
                        default= "/home/tom/Bureau/annotation/cell_type_annotation/all_anotation/pred_vs_annotation/",
                        help='path_to_project')
    
    parser.add_argument('-pa',"--path_to_annotation", type=str, 
                        default= "/home/tom/Bureau/annotation/cell_type_annotation/all_anotation/",
                        help='path_to_project')

    ###cellpose arg

    parser.add_argument('-er',"--erase_solitary", type=int, default=1, help='')

    parser.add_argument("--spot_detection", type=int, default=0, help='')
    parser.add_argument("--save_plot", type=int, default=1, help='')
    parser.add_argument("--clustering", type=int, default=1, help='')
    parser.add_argument("--epsi_cluster_cy3", type=int, default=32, help='')
    parser.add_argument("--epsi_cluster_cy5", type=int, default=32, help='')
    
    parser.add_argument("--epsi_alphashape_cy3", type=int, default=25, help='')
    parser.add_argument("--epsi_alphashape_cy5", type=int, default=25, help='')

    parser.add_argument("--overlapping_cy3", type=float, default=0.42, help='')
    parser.add_argument("--overlapping_cy5", type=float, default=0.42, help='')
    
    parser.add_argument("--remove_overlaping", type=int, default=1, help='')

    parser.add_argument("--kk_568", type=int, default = 3)    
    parser.add_argument("--kk_647", type=int, default = 3)    
    args = parser.parse_args()
    print(args)
    path_to_czi = args.path_to_czi_folder
    path_to_dapi = args.path_to_czi_folder + "tiff_data/" + "dapi/"
    path_to_af647 = args.path_to_czi_folder + "tiff_data/" + "af647/"
    path_to_af568 = args.path_to_czi_folder + "tiff_data/" + "af568/"
    path_output_segmentaton = args.path_to_czi_folder + "tiff_data/" + "predicted_mask_dapi/"
    onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

    ###############
    # Spot detection
    ###############
    if args.spot_detection:
         print("spotdetection")
         if not os.path.exists(args.path_to_project + "detected_spot_3d"+"/"):
              os.mkdir(args.path_to_project + "detected_spot_3d"+"/")
     
         dico_threshold = spot_detection_for_clustering(sigma = (1.25, 1.25, 1.25), float_out= False,
                                       rna_path = [path_to_af568+'AF568_'],
                                       path_output_segmentaton = path_output_segmentaton,
                                  threshold_input = dico_cy3,
                                  output_file = args.path_to_project + "detected_spot_3d"+"/",)
         np.save(path_to_af568+'AF568.npy', dico_threshold)
         print(dico_threshold)
         dico_threshold = spot_detection_for_clustering(sigma = (1.35, 1.35, 1.35), float_out= True,
                                       rna_path = [path_to_af647+'AF647_'],
                                       path_output_segmentaton = path_output_segmentaton,
                                  threshold_input = dico_cy5,
                                  output_file = args.path_to_project + "detected_spot_3d"+"/",)
         np.save(path_to_af647+'AF647.npy', dico_threshold)
         print(dico_threshold)
   


    if args.save_plot:
            print("ploting")
            if not os.path.exists(args.ptvplo + "plot_clustering_artifact_sup/"):
                 os.mkdir(args.ptvplo + "plot_clustering_artifact_sup/")
    
            colors = np.zeros((4,1))
            colors[0,0] = 0.12 #orange
            colors[1,0] = 0.52 #LIGHT BLUE
            colors[2,0] = 0.33 #green
            colors[3,0] = 0.85 #purple
            dico_stat = {}
            list_convex_hull_criteria = []
            for f in onlyfiles[:]:
                if f[:-5] not in onlyfiles_ano2:
                    continue
                
                print(f[:-5])
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
    
    
                if not os.path.exists(args.ptvplo + "plot_clustering_artifact_sup/" + f[:-5] +"/"):
                    os.mkdir(args.ptvplo + "plot_clustering_artifact_sup/"+ f[:-5] +"/")
                print(f)
                img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f)
                if  args.erase_solitary:
                    img_dapi_mask = erase_solitary(img_dapi_mask)
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
                    
                labels_568 = computer_optics_cluster(spots_568, eps=args.epsi_cluster_cy3, min_samples = 4, min_cluster_size=4, xi=0.05)
                labels_647 = computer_optics_cluster(spots_647, eps=args.epsi_cluster_cy5, min_samples = 4, min_cluster_size=4, xi=0.05)
    
    
                if img_dapi_mask.ndim == 2:
                    nuclei_568_0 = cluster_in_nuclei(labels_568, spots_568, img_dapi_mask, nucleus_threshold = 3) 
                    nuclei_647_0 = cluster_in_nuclei(labels_647, spots_647, img_dapi_mask, nucleus_threshold = 3) 
                    nuclei_568_1 = cluster_over_nuclei(labels_568, spots_568, img_dapi_mask, iou_threshold = 0.5)
                    nuclei_647_1 = cluster_over_nuclei(labels_647, spots_647, img_dapi_mask, iou_threshold = 0.5)
                    m, green, yellow, blue, purple =mask_image_to_rgb2D_from_list(img, img_dapi_mask, nuclei_568_1, nuclei_647_1, colors)
                if img_dapi_mask.ndim == 3:
                   
                    nuclei_568_1, positive_cluster_568,  negative_cluster_568 = cluster_over_nuclei_3D_convex_hull(labels_568, 
                                                                                            spots_568, img_dapi_mask, 
                                                                                            iou_threshold = args.overlapping_cy3)
                    nuclei_647_1, positive_cluster_647,  negative_cluster_647 =  cluster_over_nuclei_3D_convex_hull(labels_647, spots_647, 
                                                                                             img_dapi_mask, iou_threshold = args.overlapping_cy5)
                  
                    
                    m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list(np.amax(img,0),
                                                    np.amax(img_dapi_mask,0), nuclei_568_1, nuclei_647_1, colors)
                    
                    nb_no_rna = len(np.unique(img_dapi_mask)) - len(set(nuclei_647_1).union(set(nuclei_568_1)))
                    nb_cy3 = len(set(nuclei_568_1)-set(nuclei_647_1))
                    nb_cy5 = len(set(nuclei_647_1)-set(nuclei_568_1))
                    nb_both = len(set(nuclei_647_1).intersection(set(nuclei_568_1)))
                    
          
                fig, ax = plt.subplots(2,2,  figsize=(40,30))
    
                
                ax[0,0].set_title(f+ "green (no rna) %s Cy3 orange %s Cy5 blue %s  uncertain Purple %s PREDICTION" % (str(nb_no_rna), 
                                                                               str(nb_cy3),
                                                                             str(nb_cy5),
                                                                             str(nb_both)), fontsize = 15)
                ax[0,0].imshow(m)
                
                dico_anno = np.load("/home/tom/Bureau/annotation/cell_type_annotation/all_anotation/" + f[:-5] +'.npy',  allow_pickle =True).item()
                nuclei_568_1 = dico_anno["af568"]
                nuclei_647_1 = dico_anno["af647"]
                
               
              
                m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list(np.amax(img,0),
                                                    np.amax(dico_anno["mask"],0),nuclei_568_1,nuclei_647_1, colors)
                
                nb_no_rna = len(np.unique(img_dapi_mask)) - len(set(nuclei_647_1).union(set(nuclei_568_1)))
                nb_cy3 = len(set(nuclei_568_1)-set(nuclei_647_1))
                nb_cy5 = len(set(nuclei_647_1)-set(nuclei_568_1))
                nb_both = len(set(nuclei_647_1).intersection(set(nuclei_568_1)))
                
                ax[0, 1].set_title(f+ "green (no rna) %s Cy3 orange %s Cy5 blue %s  uncertain Purple %s  GROUND TRUTH" % (str(nb_no_rna), 
                                                                               str(nb_cy3),
                                                                             str(nb_cy5),
                                                                             str(nb_both)), fontsize = 15)
                ax[0, 1].imshow(m)
                
                rna = tifffile.imread(path_to_af568 + 'AF568_' + f)
                ax[1, 0].set_title("Cy3", fontsize = 15)
                ax[1, 0].imshow(np.amax(rna,0))
                
                rna = tifffile.imread(path_to_af647 + 'AF647_' + f)
                ax[1, 1].set_title("Cy5", fontsize = 15)
                ax[1, 1].imshow(np.amax(rna,0))
                
                fig.savefig(args.ptvplo + "plot_clustering_artifact_sup/" +f[:-5] +"/pred_vs_annotation" )
                    
                plt.show()
    
                
                
                if len(spots_568_old) > 15000 or len(spots_647_old) > 15000:
                    for path_to_rna in [[path_to_af568 + 'AF568_', spots_568, "red"] , [path_to_af647 + 'AF647_', spots_647, "green"]]:
                        fig, ax = plt.subplots(2,1,  figsize=(35,60))
                        rna = tifffile.imread(path_to_rna[0] + f)
                        ax[0].imshow(np.amax(rna,0))
                        ax[1].imshow(np.amax(rna,0))
                        plt.show()
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
                    plt.show()
                    fig.savefig(args.ptvplo+ "plot_clustering_artifact_sup/" + f[:-5] +"/"
                                + "rnaspot_on_dapi_no_suppression_artifacts")
                    
                   
                
                
            
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
                    plt.show()
                    fig.savefig(args.ptvplo+ "plot_clustering_artifact_sup/" + f[:-5] +"/"
                                + "rnaspot_final_on_dapi")
    
    
    
    
    

            
#%%
        
                
                
def perf_from_list(list_perf):        
    mean_acc = np.mean([l[0] for l in list_perf])
    mean_recall = np.mean([l[1] for l in list_perf])
    mean_prec = np.mean([l[2] for l in list_perf])
    TP = np.sum([l[3] for l in list_perf])
    TN = np.sum([l[4] for l in list_perf])
    FP = np.sum([l[5] for l in list_perf])
    FN = np.sum([l[6] for l in list_perf])
    accuracy = (TP+TN) / (TP+TN+FN+FP)
    recall = TP /(TP+FN)
    precision = TP / (TP +FP)
    return  mean_acc,  mean_recall, mean_prec,round(accuracy,4), round(recall, 4), round(precision,4)  


def dico_perf_pd(dico_perf, epsi_list, overlap_list):
    dico = {}
    for e in epsi_list:
         dico[e] = [perf_from_list(dico_perf[(e, overlap)])[3:] for  overlap in overlap_list]
    df = pd.DataFrame.from_dict(dico)   
    return df
        

    
#%%
print(list_perf_568)
print(perf_from_list(list_perf_647))

#%%
for k in dico_perf_568.keys():
    print(k)
    print(perf_from_list(dico_perf_568[k]))
    
#%%

for k in dico_perf_647.keys():
    print(k)
    print(perf_from_list(dico_perf_647[k])) 
#%%




#%%
def main_computation(l_para):
        print(l_para)
        epsi = l_para[0]
        overlap = l_para[1]
        list_folder =  l_para[2]
        onlyfiles_ano2 = l_para[3]
        folder_save_dict = l_para[4]
        


        list_folder_project  = list_folder
        list_perf_568_ori = []
        list_perf_647_ori = []
        list_perf_568 = []
        list_perf_647 = []
        for folder_index in range(len(list_folder)):
            print(list_folder[folder_index])
            
            parser = argparse.ArgumentParser(description='test')
            ### path to datset
            parser.add_argument('-ptz',"--path_to_czi_folder", type=str,
                                default=list_folder[folder_index], help='path_to_czi folder')
            
            parser.add_argument('-ptp',"--path_to_project", type=str, 
                                default= list_folder_project[folder_index],
                                help='path_to_project')
            
            parser.add_argument('-pa',"--path_to_annotation", type=str, 
                                default= "/home/tom/Bureau/annotation/cell_type_annotation/",
                                help='path_to_project')
        
            ###cellpose arg
        
            parser.add_argument('-er',"--erase_solitary", type=int, default=1, help='')
        
            parser.add_argument("--spot_detection", type=int, default=0, help='')
            parser.add_argument("--save_plot", type=int, default=1, help='')
            parser.add_argument("--clustering", type=int, default=1, help='')
          
            
            
            parser.add_argument("--epsi_alphashape_cy3", type=int, default=25, help='')
            parser.add_argument("--epsi_alphashape_cy5", type=int, default=25, help='')
 
            
            parser.add_argument("--remove_overlaping", type=int, default=1, help='')
        
            parser.add_argument("--kk_568", type=int, default = 3)    
            parser.add_argument("--kk_647", type=int, default = 3)    
            args = parser.parse_args()
            print(args)
            path_to_czi = args.path_to_czi_folder
            path_to_dapi = args.path_to_czi_folder + "tiff_data/" + "dapi/"
            path_to_af647 = args.path_to_czi_folder + "tiff_data/" + "af647/"
            path_to_af568 = args.path_to_czi_folder + "tiff_data/" + "af568/"
            path_output_segmentaton = args.path_to_czi_folder + "tiff_data/" + "predicted_mask_dapi/"
            onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
            onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

            ###############
            # Spot detection
            ###############
            if args.spot_detection:
                 print("spotdetection")
                 if not os.path.exists(args.path_to_project + "detected_spot_3d"+"/"):
                      os.mkdir(args.path_to_project + "detected_spot_3d"+"/")
             
                 dico_threshold = spot_detection_for_clustering(sigma = (1.25, 1.25, 1.25), float_out= False,
                                               rna_path = [path_to_af568+'AF568_'],
                                               path_output_segmentaton = path_output_segmentaton,
                                          threshold_input = {},
                                          output_file = args.path_to_project + "detected_spot_3d"+"/",)
                 np.save(path_to_af568+'AF568.npy', dico_threshold)
                 print(dico_threshold)
                 dico_threshold = spot_detection_for_clustering(sigma = (1.35, 1.35, 1.35), float_out= True,
                                               rna_path = [path_to_af647+'AF647_'],
                                               path_output_segmentaton = path_output_segmentaton,
                                          threshold_input = {},
                                          output_file = args.path_to_project + "detected_spot_3d"+"/",)
                 np.save(path_to_af647+'AF647.npy', dico_threshold)
                 print(dico_threshold)
   


            if args.save_plot:
                    print("ploting")
                    if not os.path.exists( args.path_to_project + "plot_clustering_artifact_sup/"):
                         os.mkdir(args.path_to_project + "plot_clustering_artifact_sup/")
            
                    colors = np.zeros((4,1))
                    colors[0,0] = 0.12 #orange
                    colors[1,0] = 0.52 #LIGHT BLUE
                    colors[2,0] = 0.33 #green
                    colors[3,0] = 0.85 #purple
                    for f in onlyfiles[:]:
                        if f[:-5] not in onlyfiles_ano2:
                            continue
                        print(f)
                        print()
                        if not os.path.exists( args.path_to_project + "plot_clustering_artifact_sup/" + f[:-5] +"/"):
                            os.mkdir(args.path_to_project + "plot_clustering_artifact_sup/"+ f[:-5] +"/")
                        print(f)
                        img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f)
                        if  args.erase_solitary:
                            img_dapi_mask = erase_solitary(img_dapi_mask)
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
                            spots_568 = erase_point_in_cluster_2Dalphashape(new_spots_568, removed_spots_568, 
                                                                            eps=args.epsi_alphashape_cy3, 
                                                                   min_samples = 4, min_cluster_size=4, xi=0.05)
                            spots_647 = erase_point_in_cluster_2Dalphashape(new_spots_647, removed_spots_647,
                                                                            eps=args.epsi_alphashape_cy5, 
                                                               min_samples = 4, min_cluster_size=4, xi=0.05)
                            spots_568, spots_647 = np.array(spots_568) , np.array(spots_647)
                        if img_dapi_mask.ndim == 2:
                            spots_568 = np.array([[s[1],s[2]] for s in list(spots_568)])
                            spots_647 = np.array([[s[1],s[2]] for s in list(spots_647)])
                            
                        labels_568 = computer_optics_cluster(spots_568, eps= epsi,
                                                             min_samples = 4, min_cluster_size=4, xi=0.05)
                        labels_647 = computer_optics_cluster(spots_647, eps = epsi, 
                                                             min_samples = 4, min_cluster_size=4, xi=0.05)
            
            
                        if img_dapi_mask.ndim == 2:
                            nuclei_568_0 = cluster_in_nuclei(labels_568, spots_568, img_dapi_mask, nucleus_threshold = 3) 
                            nuclei_647_0 = cluster_in_nuclei(labels_647, spots_647, img_dapi_mask, nucleus_threshold = 3) 
                            nuclei_568_1 = cluster_over_nuclei(labels_568, spots_568, img_dapi_mask, iou_threshold = 0.5)
                            #nuclei_568  = nuclei_568_0 + nuclei_568_1 
                            nuclei_647_1 = cluster_over_nuclei(labels_647, spots_647, img_dapi_mask, iou_threshold = 0.5)
                            #nuclei_647  = nuclei_647_0 + nuclei_647_1 
                            m, green, yellow, blue, purple =mask_image_to_rgb2D_from_list(img, img_dapi_mask, nuclei_568_1, nuclei_647_1, colors)
                        if img_dapi_mask.ndim == 3:

                            nuclei_568_1, positive_cluster_568,  negative_cluster_568 = cluster_over_nuclei_3D_convex_hull(labels_568, 
                                                                                                    spots_568, img_dapi_mask, 
                                                                                                    iou_threshold = overlap)
                            nuclei_647_1, positive_cluster_647,  negative_cluster_647 =  cluster_over_nuclei_3D_convex_hull(labels_647, spots_647, 
                                                                                                     img_dapi_mask, iou_threshold = overlap)
                          
                            
                            m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list(np.amax(img,0),
                                                            np.amax(img_dapi_mask,0), nuclei_568_1, nuclei_647_1, colors)
                            

                    
                        dico_anno = np.load("/home/tom/Bureau/annotation/cell_type_annotation/" + f[:-5] +'.npy',  allow_pickle =True).item()
                        true_nuclei_568 = dico_anno["af568"]
                        true_nuclei_647 = dico_anno["af647"]
                        
      
                            
                        plt.show()

                        def compute_perf(true_nuclei, nuclei_pred, img_dapi_mask):
                            TP = len(set(true_nuclei).intersection(set(nuclei_pred)))
                            pred_negative = set(np.unique(img_dapi_mask)) - set(nuclei_pred)
                            gt_negative  = set(np.unique(img_dapi_mask)) - set(true_nuclei)
                            TN =  len(gt_negative.intersection(pred_negative))
                            FP =  len(set(nuclei_pred) - set(true_nuclei))
                            FN = len(pred_negative - gt_negative)
                        
                        
                            accuracy = (TP+TN) / len(np.unique(img_dapi_mask))
                            recall = TP /(TP+FN) if (TP+FN) > 0 else 0
                            precision = TP / (TP +FP) if (TP +FP) > 0 else 0
                            return [accuracy, recall, precision, TP,TN, FP, FN]
                        perf_568 = compute_perf(true_nuclei_568, nuclei_568_1, img_dapi_mask)
                        print(perf_568)
                        perf_647 = compute_perf(true_nuclei_647, nuclei_647_1, img_dapi_mask)
                        print(perf_647)
                        list_perf_568.append(perf_568)
                        list_perf_647.append(perf_647)
                        list_perf_568_ori.append([true_nuclei_568, nuclei_568_1, np.unique(img_dapi_mask)])
                        list_perf_647_ori.append([true_nuclei_647, nuclei_647_1, np.unique(img_dapi_mask)])
        print("SAVE")
        
        
        np.save(folder_save_dict + str((epsi, overlap)), 
          [(epsi, overlap), list_perf_568, list_perf_647, list_perf_568_ori, list_perf_647_ori])

#%%


list_folder = [
              "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/"]

#list_folder = ["/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/"]
"""onlyfiles_ano2 = ['10_IR5M_Ptprb-Cy3_Mki67-Cy5_02',
                  "09_NI_Ptprb-Cy3_Mki67-Cy5_03",
                  "12_IR5M_Cap-Cy3_Mki67-Cy5_008",
                  '12_IR5M_Cap-Cy3_Mki67-Cy5_009',]"""
    
"""onlyfiles_ano2 = ["03_IR5M2201()_Pecam1-Cy5_Apln-Cy3_02",
    "01_NI2323()_Pecam1-Cy5_Apln-Cy3_03",
    "03_IR5M2201()_Pecam1-Cy5_Apln-Cy3_05"]"""

onlyfiles_ano2 = ["12_IR5M_Ptprb-Cy3_Serpine1-Cy5_04",
                  "08_IR5M_Fibin-Cy3_Serpine1-Cy5_01",
                  "08_IR5M_Fibin-Cy3_Serpine1-Cy5_14"
                  "14_IR5M_Cap-Cy3_Serpine1-Cy5_011",
                          "04_IR5M_Chil3-Cy3_Serpine1-Cy5_02",
                          "03_NI_Chil3-Cy3_Serpine1-Cy5_01"]

onlyfiles_ano2 = ['01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_01',
                          '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_02',
                          '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_04',
                          '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_06',
                          '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_07',
                          '01_NI1225_Lamp3-Cy5_Pdgfra-Cy3_08',
                          '05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_01',]

"""'05_IR5M1250_Lamp3-Cy5_Pdgfra-Cy3_mid_03']
                           "02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_006",
                           "02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_007",
                            "02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_008",
                            "02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_009",
                            "02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_012",]"""

folder = "/home/tom/Bureau/annotation/grid_search_Lamp3-Cy5_Pdgfra-Cy3/"
#%%
import multiprocessing      
l_params = []
l_params = []
epsilon_list = [ 15, 20, 25, 30, 32, 34, 35, 40, 45]
overlap = [ 0.30,0.35, 0.38, 0.4, 0.42, 0.45, 0.50, 0.55, 0.6]
for e in epsilon_list :
    for o in overlap:
        l_params.append([e,o, list_folder, onlyfiles_ano2, folder ])
#%%
number_processes = 12
pool = multiprocessing.Pool(number_processes)
results = pool.map_async(main_computation, l_params)
pool.close()
pool.join()
#main_computation(l_params[0])
#%%
dico_perf_568 = {}
dico_perf_647 = {}
dico_perf_input_568 = {}
dico_perf_input_647 = {}
for e in  epsilon_list:
    for o in overlap :
        l = np.load(folder+ str((e, o))+".npy")
        dico_perf_568[(e, o)] = l[1]
        dico_perf_647[(e, o)] = l[2]
        dico_perf_input_568[(e, o)]= l[3]
        dico_perf_input_647[(e, o)] = l[4]
        
#%%
df_568 =dico_perf_pd(dico_perf_568, epsi_list  = epsilon_list, overlap_list = overlap)

df_647 =dico_perf_pd(dico_perf_647, epsi_list  = epsilon_list, overlap_list = overlap)

from tabulate import tabulate
print(tabulate(df_568, headers='keys', tablefmt='psql'))

df_568.to_pickle(folder + "result_CY3.pkl")

df_647.to_pickle(folder  + "result_CY5.pkl")


#%%

#%%
    

def dico_perf_pd2(dico_perf, epsi_list, overlap_list):
    dico = {}
    for e in epsi_list:
         dico[e] = [[round(dico_perf[(e, overlap)][0], 4),round(dico_perf[(e, overlap)][1], 4), 
                     round(dico_perf[(e, overlap)][2], 4)] for  overlap in overlap_list]
    df = pd.DataFrame.from_dict(dico)   
    return df       
        

#%%
def precompute_balanced_input(dico_perf_input):
    dico_perf_input_res = {}
    for key in dico_perf_input.keys():
        current_param = dico_perf_input[key]
        list_unique = current_param[0][-1]
        
        print(current_param[0][0])
        
        offset = 0
        gt_positive = []
        prediction = []
        all_nuclei = []
        gt_negative =[]
        for i in range(len(current_param)):
            #print((np.max(list_unique[i]), current_param[i][-1]))
            
            gt_positive += list(np.array(current_param[i][0]) + offset)
            prediction += list(np.array(current_param[i][1]) + offset)
            gt_negative  += list(np.array(list(set(list_unique) - set(current_param[i][0]))) + offset)
            all_nuclei += list(np.array(list_unique) + offset)
            offset += np.max(list_unique[i])
            
        # selected balanced dataset
        accuracy = 0
        recall = 0
        precision = 0
        number_of_sample = 10
        for nimporte in range(number_of_sample):
            nb_to_select = min([len(gt_positive), len(gt_negative)])
            print(nb_to_select)
            
            gt_positive_sample = set(random.sample(gt_positive, nb_to_select))
            gt_negative_sample = set(random.sample(gt_negative, nb_to_select))
            prediction_sample = set(prediction).intersection(set(gt_negative_sample).union(set(gt_positive_sample)))
            all_nuclei_sample = set(all_nuclei).intersection(set(gt_negative_sample).union(set(gt_positive_sample)))
            
            TP = len(set(gt_positive_sample).intersection(prediction_sample))
            pred_negative = all_nuclei_sample -  prediction_sample
            TN =  len(gt_negative_sample.intersection(pred_negative))
            FP =  len(prediction_sample - gt_positive_sample)
            FN = len(pred_negative - gt_negative_sample)
        
        
            accuracy += (TP+TN) / len(set(gt_negative_sample).union(set(gt_positive_sample))) if len(set(gt_negative_sample).union(set(gt_positive_sample))) else 0
            recall += TP /(TP+FN) if (TP+FN) > 0 else 0
            precision += TP / (TP +FP) if (TP +FP) > 0 else 0  
        dico_perf_input_res[key] = [accuracy/number_of_sample , recall/number_of_sample , precision/number_of_sample ]
    return dico_perf_input_res
        
 
dico_perf_input_res_568 = precompute_balanced_input(dico_perf_input_568)
dico_perf_input_res_647 = precompute_balanced_input(dico_perf_input_647)


df_568 =dico_perf_pd2(dico_perf_input_res_568, epsi_list  = epsilon_list, 
                     overlap_list = overlap)


df_647 =dico_perf_pd2(dico_perf_input_res_647 , epsi_list  = epsilon_list
                     , overlap_list = overlap)

from tabulate import tabulate
print(tabulate(df_568, headers='keys', tablefmt='psql'))

df_568.to_pickle(folder  + "result_balanced_CY3.pkl")

df_647.to_pickle(folder +"result_balanced_CY5.pkl")