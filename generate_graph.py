# -*- coding: utf-8 -*-

#%%
import argparse

import time
import os
from os import listdir
from os.path import isfile, join

import tifffile
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import networkx as nx
import community
import matplotlib.cm as cm
from spots.plot import  hsv_to_rgb


from spots.post_processing import erase_solitary

from graph.graph_general_statistic import get_neighbors_empty
from graph.graph_construction import label_with_empty, get_adjacent_list, remove_long_edge, get_dico_centroid, remove_self_loop_adlist, remove_self_loop_dico
from graph.community import  get_louvain_partition
#%%


def plot_graph(img_dapi_mask, adjacent_list, indice=10, labels_with_empty=None, f=None):
    """

    Parameters
    ----------
    img_dapi_mask
    adjacent_list
    indice
    empty_space
    f

    Returns
    -------

    """
    if indice is not None:
        dico_centroid = get_dico_centroid(img_dapi_mask[indice])
    else:
        dico_centroid = get_dico_centroid(img_dapi_mask)

    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    if f is not None:
        fig.suptitle(f, fontsize=23)
    ccc = img_dapi_mask
    if labels_with_empty is not None:
        ax.imshow(labels_with_empty[indice], cmap='nipy_spectral', alpha=0.5)
    if indice is not None:
        ax.imshow(ccc[indice], cmap='nipy_spectral', alpha=0.5)
    else:
        ax.imshow(np.amax(ccc, 0), cmap='nipy_spectral', alpha=0.5)
    for edge in adjacent_list:
        if edge[0] in set(dico_centroid.keys()) and edge[1] in set(dico_centroid.keys()):
            point1 = dico_centroid[edge[-2]]
            point2 = dico_centroid[edge[-1]]
            x_values = [point1[-1], point2[-1]]
            y_values = [point1[-2], point2[-2]]
            ax.plot(x_values, y_values, color='green', linewidth=5)
    plt.show()


def plot_nuclei_plus_voronoid(img_dapi_mask, labels_with_empty, indice=10):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(labels_with_empty[indice], cmap='nipy_spectral', alpha=0.5)
    ax.imshow(img_dapi_mask[indice], cmap='nipy_spectral', alpha=0.5)
    plt.show()


def plot_commnunities(dapi, masks, partition):
    colors = np.zeros((max(partition.values()),1))
    for i_par in range(len(colors)):
        colors[i_par, 0] = 1/len(colors) * i_par
    if dapi.ndim>2:
        dapi = np.amax(dapi, 0).astype(np.float32)
    else:
        dapi = dapi.astype(np.float32)
    if masks.ndim == 3:
        masks = np.amax(masks, 0)
    dapi -= dapi.min()
    dapi /= dapi.max()
    HSV = np.zeros((dapi.shape[0], dapi.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(dapi*1.5, 0, 1.0)


    for n in np.unique(masks):
        if n==0:
            continue
        ipix = (masks==n).nonzero()
        HSV[ipix[0],ipix[1],0] = colors[partition[n]-1,0]
        HSV[ipix[0],ipix[1],1] = 1.0

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8) #
    return RGB
#%%
def plot_commnunities_plus_edge(dapi, masks, partition,
                                adjacent_list, dico_centroid = None,
                                title = 'None',
                                rao_stirling_index_nuclei = None):


    if dico_centroid is None:
        dico_centroid = get_dico_centroid(masks)

    colors = np.zeros((max(partition.values()) +1,1))
    for i_par in range(len(colors)):
        colors[i_par, 0] =  1/len(colors) * i_par
    if dapi.ndim>2:
        dapi = np.amax(dapi, 0).astype(np.float32)
    else:
        dapi = dapi.astype(np.float32)
    if masks.ndim == 3:
        masks = np.amax(masks, 0)
    dapi -= dapi.min()
    dapi /= dapi.max()
    HSV = np.zeros((dapi.shape[0], dapi.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(dapi*1.5, 0, 1.0)

    unique_nuclei  = np.unique(masks)
    for n in np.unique(masks):
        if n==0:
            continue
        try:
            if rao_stirling_index_nuclei is not None:
                if n in rao_stirling_index_nuclei:
                    ipix = ndi.maximum_filter((masks==n).astype(int), 20).nonzero()
                    HSV[ipix[0], ipix[1], 0] = 0.11 # green
                    HSV[ipix[0], ipix[1], 1] = 1.0

            ipix = (masks == n).nonzero()
            HSV[ipix[0],ipix[1],0] = colors[partition[n]-1,0]
            HSV[ipix[0],ipix[1],1] = 1.0



        except Exception as e:
            ipix = (masks==n).nonzero()
            HSV[ipix[0],ipix[1],0] = colors[max(partition.values()),0]
            HSV[ipix[0],ipix[1],1] = 1.0
            print(e)
            print("the node %s is not in any partition" % str(n))

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8) #
    
    fig, ax =  plt.subplots(1,1, figsize = (10, 10))
    ax.imshow(RGB)
    for edge in adjacent_list:
        if edge[0] in set(unique_nuclei) and edge[1] in set(unique_nuclei):
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            x_values = [point1[-1], point2[-1]]
            y_values = [point1[-2], point2[-2]]
            ax.plot(x_values, y_values, color='green', linewidth=1)
    ax.set_title(title)
    plt.show()
    return RGB, fig, ax



def plot_top_plus_edge(dapi, masks,dico_centrality,  adjacent_list, dico_centroid = None, top = 10) :
    """

    Parameters
    ----------
    dapi
    masks
    dico_centrality: dico with the 'top' metrics to plot with key=Node index, values = metric
    adjacent_list
    dico_centroid
    top

    Returns
    -------

    """
    if dico_centroid is None:
        dico_centroid = get_dico_centroid(masks)
        
    kv = [ (k, dico_centrality[k]) for k in dico_centrality]
    kv = sorted(kv, key=lambda tup: tup[1])
    
    top_centrale_nodes = kv[-top:]
    top_centrale_nodes = [t[0]for t in top_centrale_nodes]
    print(top_centrale_nodes)

    colors = np.zeros([2,1])
    colors[0,0] = 0.33 #green cy3
    colors[1,0] = 0.61 # red cy5
    for i_par in range(len(colors)):
        colors[i_par, 0] =  1/len(colors) * i_par
    if dapi.ndim>2:
        dapi = np.amax(dapi, 0).astype(np.float32)
    else:
        dapi = dapi.astype(np.float32)
    if masks.ndim == 3:
        masks = np.amax(masks, 0)
    dapi -= dapi.min()
    dapi /= dapi.max()
    HSV = np.zeros((dapi.shape[0], dapi.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(dapi*1.5, 0, 1.0)

    unique_nuclei  = np.unique(masks)
    for n in np.unique(masks):
        if n==0:
            continue
        ipix = (masks==n).nonzero()
        if n in top_centrale_nodes:
            HSV[ipix[0],ipix[1],0] = colors[0,0]
        else:
            HSV[ipix[0],ipix[1],0] = colors[1,0]
        HSV[ipix[0],ipix[1],1] = 1.0

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8) #
    
    fig, ax =  plt.subplots(1,1, figsize = (10, 10))
    ax.imshow(RGB)
    for edge in adjacent_list:
        if edge[0] in set(unique_nuclei) and edge[1] in set(unique_nuclei):
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            x_values = [point1[-1], point2[-1]]
            y_values = [point1[-2], point2[-2]]
            ax.plot(x_values, y_values, color='green', linewidth=1)
    plt.show()
    return RGB, top_centrale_nodes

    def plot_top_h_centrality_plus_edge(dapi, masks, dico_centrality, adjacent_list, centroid = None, top = 10) :
        if centroid is not None:
            pass
        else:
            centroid = get_dico_centroid(masks)

        kv = [ (k, dico_centrality[k]) for k in dico_centrality]
        kv = sorted(kv, key=lambda tup: tup[1])
        
        top_centrale_nodes = kv[-top:]
        top_centrale_nodes = [t[0]for t in top_centrale_nodes]
        print(top_centrale_nodes)
    
        colors = np.zeros([2,1])
        colors[0,0] = 0.33 #green cy3
        colors[1,0] = 0.61 # red cy5
        for i_par in range(len(colors)):
            colors[i_par, 0] =  1/len(colors) * i_par
        if dapi.ndim>2:
            dapi = np.amax(dapi, 0).astype(np.float32)
        else:
            dapi = dapi.astype(np.float32)
        if masks.ndim == 3:
            masks = np.amax(masks, 0)
        dapi -= dapi.min()
        dapi /= dapi.max()
        HSV = np.zeros((dapi.shape[0], dapi.shape[1], 3), np.float32)
        HSV[:,:,2] = np.clip(dapi*1.5, 0, 1.0)
    
        unique_nuclei  = np.unique(masks)
        for n in np.unique(masks):
            if n==0:
                continue
            ipix = (masks==n).nonzero()
            if n in top_centrale_nodes:
                HSV[ipix[0],ipix[1],0] = colors[0,0]
            else:
                HSV[ipix[0],ipix[1],0] = colors[1,0]
            HSV[ipix[0],ipix[1],1] = 1.0
    
        RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8) #
        
        fig, ax =  plt.subplots(1,1, figsize = (10, 10))
        ax.imshow(RGB)
        for edge in adjacent_list:
            if edge[0] in set(unique_nuclei) and edge[1] in set(unique_nuclei):
                point1 = centroid[edge[0]]
                point2 = centroid[edge[1]]
                x_values = [point1[-1], point2[-1]]
                y_values = [point1[-2], point2[-2]]
                ax.plot(x_values, y_values, color='green', linewidth=1)
        plt.show()
        return RGB, top_centrale_nodes

    def plot_one_type_plus_edge(dapi, masks, list_type_plus, adjacent_list, centroid=None):
        if centroid is not None:
            pass
        else:
            centroid = get_dico_centroid(masks)


        top_centrale_nodes = list_type_plus
        print(top_centrale_nodes)

        colors = np.zeros([2, 1])
        colors[0, 0] = 0.33  # green cy3
        colors[1, 0] = 0.61  # red cy5
        for i_par in range(len(colors)):
            colors[i_par, 0] = 1 / len(colors) * i_par
        if dapi.ndim > 2:
            dapi = np.amax(dapi, 0).astype(np.float32)
        else:
            dapi = dapi.astype(np.float32)
        if masks.ndim == 3:
            masks = np.amax(masks, 0)
        dapi -= dapi.min()
        dapi /= dapi.max()
        HSV = np.zeros((dapi.shape[0], dapi.shape[1], 3), np.float32)
        HSV[:, :, 2] = np.clip(dapi * 1.5, 0, 1.0)

        unique_nuclei = np.unique(masks)
        for n in np.unique(masks):
            if n == 0:
                continue
            ipix = (masks == n).nonzero()
            if n in top_centrale_nodes:
                HSV[ipix[0], ipix[1], 0] = colors[0, 0]
            else:
                HSV[ipix[0], ipix[1], 0] = colors[1, 0]
            HSV[ipix[0], ipix[1], 1] = 1.0

        RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)  #

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(RGB)
        for edge in adjacent_list:
            if edge[0] in set(unique_nuclei) and edge[1] in set(unique_nuclei):
                point1 = centroid[edge[0]]
                point2 = centroid[edge[1]]
                x_values = [point1[-1], point2[-1]]
                y_values = [point1[-2], point2[-2]]
                ax.plot(x_values, y_values, color='green', linewidth=1)
        plt.show()
        return RGB, centroid



def compute_all_adjancy_list(folder, threshold_empty=5, name_save = "2907",
                             heuristic = "double_dapi", get_empty =False):
    path_output_segmentaton = folder + "tiff_data/predicted_mask_dapi/"
    path_to_af568 = folder + "tiff_data/" + "af568/"
    path_to_af647 = folder + "tiff_data/" + "af647/"

    if not os.path.exists(folder + "adjacent_list/"):
        os.mkdir(folder + "adjacent_list/")
    if not os.path.exists(folder + "labels_with_empty/"):
        os.mkdir(folder + "labels_with_empty/")
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
                 isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

    for f in onlyfiles:
        t = time.time()
        print(f)
        if os.path.isfile(folder + "adjacent_list/" + "ad_list" + name_save + f +".npy"):
            continue
        img_dapi_mask = erase_solitary(tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f))
        af568 = tifffile.imread(path_to_af568 + "AF568_" + f)
        af647 = tifffile.imread(path_to_af647 + "AF647_" + f)
        labels_with_empty = label_with_empty(img_dapi_mask, af568,
                                             af647,
                                             threshold_percent=threshold_empty,
                                             heuristic=heuristic)
        print("labels_with_empty shape %s" % str(labels_with_empty.shape))
        if not get_empty:
            adjacent_list, dico_ngb = get_adjacent_list(img_dapi_mask, labels_with_empty)
            np.save(folder + "adjacent_list/" + "ad_list" + name_save + f, np.array(adjacent_list))
            np.save(folder + "adjacent_list/" + "dico_no" + name_save + f, dico_ngb)
            np.save(folder + "labels_with_empty/" + "lwe_" + name_save + f, labels_with_empty)
        else: ##this part is useless....
            contact_wt_empty = get_neighbors_empty(labels_with_empty, max_filter_vacumm = 7, max_filter_frontier = 8)
            np.save(folder + "adjacent_list/" + "dico_empty" + name_save + f, contact_wt_empty)
        print("time for one images %s" % str(time.time() - t))


#%% script zone
if __name__ == "__main__":
#%%
    folder_list = [#"/media/tom/Elements/to_take/210205_Prolicence/aCap prolif/",
                   "/media/tom/Elements/to_take/210205_Prolicence/aCap senes/",
                   "/media/tom/Elements/to_take/210205_Prolicence/gCap prolif/",
                   "/media/tom/Elements/to_take/210205_Prolicence/gCap senes/"]

    for folder in folder_list:
        compute_all_adjancy_list(folder, threshold_empty=5, name_save="dd_t5_2907",
                             heuristic="double_dapi")













#%%
    parser = argparse.ArgumentParser(description='generate_graph')
    parser.add_argument('-ptt', "--path_to_take", type=str,
                        default="/media/tom/Elements/to_take/",
                        help='path_to_take')

    parser.add_argument('-ptf', "--path_to_folder", type=str,
                        default="201127_AM_fibro/",
                        help='path_to_take')

    parser.add_argument("--threshold_empty", type=int,
                        default=5,
                        help='path_to_take')

    parser.add_argument("--name_save", type=str,
                        default="2608",
                        help='path_to_take')

    parser.add_argument("--heuristic", type=str,
                        default="double_dapi",
                        help='path_to_take')

    parser.add_argument("--compute_empty_neighbors", type=int, default=1)
    parser.add_argument("--port", default=39949)
    parser.add_argument("--mode", default='client')
    args = parser.parse_args()
    print(args)

    full_path_folder = args.path_to_take + args.path_to_folder
    threshold_empty = args.threshold_empty
    name_save = args.name_save
    heuristic = args.heuristic

    path_output_segmentaton = full_path_folder + "tiff_data/predicted_mask_dapi/"
    path_to_af568 = full_path_folder + "tiff_data/" + "af568/"
    path_to_af647 = full_path_folder + "tiff_data/" + "af647/"
    path_to_dapi = full_path_folder + "tiff_data/" + "dapi/"
    if not os.path.exists(full_path_folder + "adjacent_list/"):
        os.mkdir(full_path_folder + "adjacent_list/")
    if not os.path.exists(full_path_folder + "labels_with_empty/"):
        os.mkdir(full_path_folder + "labels_with_empty/")
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
                 isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
    dico_partition  = np.load(full_path_folder + "dico_partition2408.npy", allow_pickle = True).item()
    dico_rao_stirling = np.load(full_path_folder + '/dico_rao_stirling.npy',
                                allow_pickle=True).item()

    for f in onlyfiles:
        t = time.time()

        print(f)
        img_dapi_mask = erase_solitary(tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f))
        af568 = tifffile.imread(path_to_af568 + "AF568_" + f)
        af647 = tifffile.imread(path_to_af647 + "AF647_" + f)
        dapi = tifffile.imread(path_to_dapi + "dapi_" + f)
        partition = dico_partition[f][0][1][0]

        rao_stirling = dico_rao_stirling[f]


        ad_list = remove_self_loop_adlist(np.load(full_path_folder + "adjacent_list/" + "ad_list2405_wt0" + f + ".npy"))

        kv = [(k, rao_stirling[k]) for k in rao_stirling]
        kv = sorted(kv, key=lambda tup: tup[1])
        rao_stirling_index_nuclei = kv[-6:]
        rao_stirling_index_nuclei_bis = [i[0] for i in rao_stirling_index_nuclei]
        print(rao_stirling_index_nuclei)

        rgb, fig, ax = plot_commnunities_plus_edge(dapi, img_dapi_mask, partition, ad_list,
                                                   dico_centroid=None, title = f,
                                                   rao_stirling_index_nuclei=rao_stirling_index_nuclei_bis)
        fig.savefig("/home/tom/Bureau/phd/first_lustra/plot_rao/" +f)
        plt.show()

    #%%
    parser = argparse.ArgumentParser(description='generate_graph')
    parser.add_argument('-ptt', "--path_to_take", type=str,
                        default="/media/tom/Elements/to_take/",
                        help='path_to_take')

    parser.add_argument('-ptf', "--path_to_folder", type=str,
                        default="201127_AM_fibro/",
                        help='path_to_take')

    parser.add_argument("--threshold_empty", type=int,
                        default=5,
                        help='path_to_take')

    parser.add_argument("--name_save", type=str,
                        default="2608",
                        help='path_to_take')

    parser.add_argument("--heuristic", type=str,
                        default="double_dapi",
                        help='path_to_take')

    parser.add_argument("--compute_empty_neighbors", type=int, default=1)
    parser.add_argument("--port", default=39949)
    parser.add_argument("--mode", default='client')
    args = parser.parse_args()
    print(args)

    full_path_folder = args.path_to_take + args.path_to_folder
    threshold_empty = args.threshold_empty
    name_save = args.name_save
    heuristic = args.heuristic

    path_output_segmentaton = full_path_folder + "tiff_data/predicted_mask_dapi/"
    path_to_af568 = full_path_folder + "tiff_data/" + "af568/"
    path_to_af647 = full_path_folder + "tiff_data/" + "af647/"
    path_to_dapi = full_path_folder + "tiff_data/" + "dapi/"
    if not os.path.exists(full_path_folder + "adjacent_list/"):
        os.mkdir(full_path_folder + "adjacent_list/")
    if not os.path.exists(full_path_folder + "labels_with_empty/"):
        os.mkdir(full_path_folder + "labels_with_empty/")
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
                 isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
    dico_partition  = np.load(full_path_folder + "dico_partition2408.npy", allow_pickle = True).item()
    dico_rao_stirling = np.load(full_path_folder + '/dico_rao_stirling.npy',
                                allow_pickle=True).item()

    for f in onlyfiles:
        t = time.time()

        print(f)
        img_dapi_mask = erase_solitary(tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f))
        af568 = tifffile.imread(path_to_af568 + "AF568_" + f)
        af647 = tifffile.imread(path_to_af647 + "AF647_" + f)
        dapi = tifffile.imread(path_to_dapi + "dapi_" + f)
        partition = dico_partition[f][0][1][0]
        ad_list = remove_self_loop_adlist(np.load(full_path_folder + "adjacent_list/" + "ad_list2405_wt0" + f + ".npy"))



        rgb, fig, ax = plot_commnunities_plus_edge(dapi, img_dapi_mask, partition, ad_list, dico_centroid=None, title = f)
        fig.savefig("/home/tom/Bureau/phd/first_lustra/plot_of_commnunities/" +f)
        plt.show()

#%%
    for f in onlyfiles:
        t = time.time()

        print(f)
        ad_list = remove_self_loop_adlist(np.load(folder + "adjacent_list/" + "ad_list2405_wt0" + f + ".npy"))

        img_dapi_mask = erase_solitary(tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f))
        af568 = tifffile.imread(path_to_af568 + "AF568_" + f)
        af647 = tifffile.imread(path_to_af647 + "AF647_" + f)
        dapi = tifffile.imread(path_to_dapi + "dapi_" + f)
        #labels_with_empty = label_with_empty(img_dapi_mask, af568,
        #                                     af647,
        #                                     threshold_percent=threshold_empty,
        #                                     heuristic=heuristic)

        dico_type = np.load("/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/dico_stat_2106.npy",
                allow_pickle=True).item()

        positive_cell = [dico_type[f][5][i][3] for i in range(len(dico_type[f][5]))]
        plt.title(f)
        plot_one_type_plus_edge(dapi, img_dapi_mask, np.unique(img_dapi_mask)[4:20], ad_list, centroid=None)
        plt.title(f)
        plt.show()