
import argparse
import tifffile
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from graph.graph_construction import remove_self_loop_dico, remove_self_loop_adlist
from graph.graph_general_statistic import  compute_nb_edge, compute_nb_empty_contact
from graph.community import compute_community_distance, nb_cell_type_per_community
from spots.post_processing import erase_solitary
from os import listdir
from os.path import isfile, join
import os
from utils import get_dye, get_mouse_name


def script_compute_nb_edge(folder, name_ad = "dd_t5_2907", name_dico_type = "dico_stat_2106.npy"):
    """
    Parameters
    ----------
    folder
    name_ad
    name_dico_type
    Returns
    -------
    """
    print(folder[-15:])

    path_output_segmentaton = folder + "tiff_data/predicted_mask_dapi/"
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
                 isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

    path_to_ad = folder + "adjacent_list/"
    dico_type = np.load(folder + name_dico_type,
                        allow_pickle=True).item()
    dico_edge = {}
    for f in onlyfiles:
        try:
            ad_list = remove_self_loop_adlist(np.load(path_to_ad + "ad_list" +name_ad + f + ".npy"))
            dico_nb = remove_self_loop_dico(np.load(path_to_ad +  "dico_no" + name_ad + f + ".npy",
                                                    allow_pickle=True).item())

            positive_af568 = [dico_type[f][5][i][3] for i in range(len(dico_type[f][5]))]
            positive_af647 = [dico_type[f][6][i][3] for i in range(len(dico_type[f][6]))]
            nb_af568, nb_af647, nb_both, degre_list_af568, degre_list_af647  = compute_nb_edge(dico_nb, positive_af568, positive_af647)
            nb_total = len(ad_list)
            dico_edge[f] = [("af568", nb_af568), ("af647", nb_af647), ("both", nb_both), ("total", nb_total), degre_list_af568, degre_list_af647]
        except Exception as e:
            print(e)
            dico_edge[f] = ["type no available ?"]
    np.save(folder + "dico_edge", dico_edge)
    return dico_edge


def compute_expected_edge(gene, folders_list, path_to_take = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/"):
in
    ### scrap the name of the image of the gene it is
    dico_NI_ratio = {}
    dico_IR5M_ratio = {}
    for folder_name in folders_list:
        path_output_segmentaton = path_to_take + folder_name + "tiff_data/" + "predicted_mask_dapi/"
        onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
        onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
        ## load edg dictionary
        dico_edge = np.load(path_to_take + folder_name + "dico_edge.npy", allow_pickle=True).item()
        sorted_name = np.sort(list(dico_edge.keys()))
        for key_cell_name in sorted_name:
            if not any(word in key_cell_name for word in gene):
                continue
            dye_type = get_dye(gene, key_cell_name)
            if len(dico_edge[key_cell_name])==1: #two images in the dataset are not computationally tractable
                print(key_cell_name)
                print(dico_edge[key_cell_name])
                continue
            ### compute the ratio (number of edge / number of expected edge)
            if dye_type == "Cy3":
                #af568
                nb_edge_af568 = dico_edge[key_cell_name][0][1]
                af568_degree_list = dico_edge[key_cell_name][4]
                total_edge = dico_edge[key_cell_name][3][1]
                expected_af568 = np.sum([[af568_degree_list[i] *
                         af568_degree_list[j] / total_edge for i in range(len(af568_degree_list))]
                        for j in range(len(af568_degree_list))])
                expected = expected_af568
                real = nb_edge_af568
            else:
                #af647
                nb_edge_af647 = dico_edge[key_cell_name][1][1]
                af647_degree_list = dico_edge[key_cell_name][5]
                total_edge = dico_edge[key_cell_name][3][1]
                expected_af647 = np.sum([[af647_degree_list[i] *
                         af647_degree_list[j] / total_edge for i in range(len(af647_degree_list))]
                        for j in range(len(af647_degree_list))])
                expected = expected_af647
                real = nb_edge_af647
            if  any(word in key_cell_name for word in ["NI", "Ctrl"]):
                dico_NI_ratio[key_cell_name] = [expected, real, real/expected]

            if  any(word in key_cell_name for word in ["IR5M"]):
                dico_IR5M_ratio[key_cell_name] = [expected, real, real/expected]

    #now save the dico :)
    if not os.path.exists(path_to_take + "stat_edge/"):
        os.mkdir(path_to_take + "stat_edge/")
    np.save(path_to_take + "stat_edge/" + gene[0] + "NI", dico_NI_ratio)
    np.save(path_to_take + "stat_edge/" + gene[0] + "IR5M", dico_IR5M_ratio)

def compute_empty_contact(gene,  folders_list, path_to_take = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/"):
    """

    Parameters
    ----------
    gene
    folders_list
    path_to_take

    Returns
    -------

    """
    ### scrap the name of the image of the gene it is
    dico_NI_ratio = {}
    dico_IR5M_ratio = {}
    for folder_name in folders_list:
        path_output_segmentaton = path_to_take + folder_name + "tiff_data/" + "predicted_mask_dapi/"
        onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
        onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
        ## load edge dictionary
        dico_empty = np.load(path_to_take + folder_name + "dico_empty.npy", allow_pickle=True).item()
        sorted_name = np.sort(list(dico_empty.keys()))
        for key_cell_name in sorted_name:
            if not any(word in key_cell_name for word in gene):
                continue
            dye_type = get_dye(gene, key_cell_name)
            if len(dico_empty[key_cell_name])==1: #two images in the dataset are not computationally tractable
                print(key_cell_name)
                print(dico_empty[key_cell_name])
                continue
            if dye_type == "Cy3":
                #af568
                percentage_empty = dico_empty[key_cell_name][0][1]

            else:
                #af647
                percentage_empty = dico_empty[key_cell_name][1][1]
            if  any(word in key_cell_name for word in ["NI", "Ctrl"]):
                dico_NI_ratio[key_cell_name] = [percentage_empty]

            if  any(word in key_cell_name for word in ["IR5M"]):
                dico_IR5M_ratio[key_cell_name] = [percentage_empty]

    #now save the dico :)
    if not os.path.exists(path_to_take + "percentage_empty/"):
        os.mkdir(path_to_take + "percentage_empty/")
    np.save(path_to_take + "percentage_empty/" + gene[0] + "NI", dico_NI_ratio)
    np.save(path_to_take + "percentage_empty/" + gene[0] + "IR5M", dico_IR5M_ratio)