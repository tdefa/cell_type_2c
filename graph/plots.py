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
from spots.plot import hsv_to_rgb

from spots.post_processing import erase_solitary

from graph.graph_construction import label_with_empty, get_adjacent_list, remove_long_edge, get_dico_centroid
from graph.graph_general_statistic import compute_centrality_betweeness, compute_centrality_harmonic, get_neighbors_empty


# %%


def plot_graph(img_dapi_mask, adjacent_list, indice=10, empty_space=None, f=None):
    if indice is not None:
        dico_centroid = get_dico_centroid(img_dapi_mask[indice])
    else:
        dico_centroid = get_dico_centroid(img_dapi_mask)

    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    if f is not None:
        fig.suptitle(f, fontsize=23)
    ccc = img_dapi_mask
    if empty_space is not None:
        ccc[empty_space == 0] = 10
    if indice is not None:
        ax.imshow(ccc[indice], cmap='gist_earth', alpha=0.5)
    else:
        ax.imshow(np.amax(ccc, 0), cmap='gist_earth', alpha=0.5)
    for edge in adjacent_list:
        if edge[0] in set(dico_centroid.keys()) and edge[1] in set(dico_centroid.keys()):
            point1 = dico_centroid[edge[-2]]
            point2 = dico_centroid[edge[-1]]
            x_values = [point1[-1], point2[-1]]
            y_values = [point1[-2], point2[-2]]
            ax.plot(x_values, y_values, color='green', linewidth=5)
    plt.show()


def plot_graph_plus_voronoid(img_dapi_mask, labels_with_empty, indice=10):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(labels_with_empty[indice], cmap='ocean', alpha=0.4)
    ax.imshow(labels_with_empty[indice], cmap='prism', alpha=0.4)

    ax.imshow(img_dapi_mask[indice], cmap='gist_earth', alpha=0.5)
    plt.show()


def plot_commnunities(dapi, masks, partition):
    colors = np.zeros((max(partition.values()), 1))
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

    for n in np.unique(masks):
        if n == 0:
            continue
        ipix = (masks == n).nonzero()
        HSV[ipix[0], ipix[1], 0] = colors[partition[n] - 1, 0]
        HSV[ipix[0], ipix[1], 1] = 1.0

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)  #
    return RGB


def plot_commnunities_plus_edge(dapi, masks, partition, adjacent_list, dico_centroid=None):
    if dico_centroid is None:
        dico_centroid = get_dico_centroid(masks)

    colors = np.zeros((max(partition.values()) + 1, 1))
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
        try:
            ipix = (masks == n).nonzero()
            HSV[ipix[0], ipix[1], 0] = colors[partition[n] - 1, 0]
            HSV[ipix[0], ipix[1], 1] = 1.0
        except Exception as e:
            ipix = (masks == n).nonzero()
            HSV[ipix[0], ipix[1], 0] = colors[max(partition.values()), 0]
            HSV[ipix[0], ipix[1], 1] = 1.0
            print(e)
            print("the node %s is not in any partition" % str(n))

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)  #

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(RGB)
    for edge in adjacent_list:
        if edge[0] in set(unique_nuclei) and edge[1] in set(unique_nuclei):
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            x_values = [point1[-1], point2[-1]]
            y_values = [point1[-2], point2[-2]]
            ax.plot(x_values, y_values, color='green', linewidth=1)
    plt.show()
    return RGB


def plot_top_h_centrality_plus_edge(dapi, masks, dico_centrality,
                                    adjacent_list, dico_centroid=None, top=10):
    """

    Parameters
    ----------
    dapi
    masks
    dico_centrality
    adjacent_list
    dico_centroid
    top

    Returns
    -------

    """
    if dico_centroid is None:
        dico_centroid = get_dico_centroid(masks)

    kv = [(k, dico_centrality[k]) for k in dico_centrality]
    kv = sorted(kv, key=lambda tup: tup[1])

    top_centrale_nodes = kv[-top:]
    top_centrale_nodes = [t[0] for t in top_centrale_nodes]
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
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            x_values = [point1[-1], point2[-1]]
            y_values = [point1[-2], point2[-2]]
            ax.plot(x_values, y_values, color='green', linewidth=1)
    plt.show()
    return RGB, top_centrale_nodes

def plot_top_h_centrality_plus_edge(dapi, masks, dico_centrality, adjacent_list, centroid=None, top=10):
    if centroid is not None:
        pass
    else:
        centroid = get_dico_centroid(masks)

    kv = [(k, dico_centrality[k]) for k in dico_centrality]
    kv = sorted(kv, key=lambda tup: tup[1])

    top_centrale_nodes = kv[-top:]
    top_centrale_nodes = [t[0] for t in top_centrale_nodes]
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
    return RGB, top_centrale_nodes
