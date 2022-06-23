

#%%
import time


import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import networkx as nx

#######
# graph construction
#######

def label_with_empty(img_dapi_mask, af568, af647, threshold_percent= 5, heuristic = "double_dapi"):
    """
    Parameters
    ----------
    img_dapi_mask : 3D segmentation mask
        af568 : numpy arry
        smfish signal


    threshold_percent : float, optional
        DESCRIPTION. The default is 40.
        between 0 and 100

    heuristic : basestring
        choice of the method to detect the empty space in the tissue
        if double_dapi, it fixs the threshold at the 5 percentile of intensity of location corresponding to the nucleus
        (todo make it more clear)
    Returns
    -------
    labels_with_empty : TYPE array of shape img_dapi_mask
        watersheld without the empty space.
    """

    if heuristic == "double_dapi":
        empty_space_568 = af568 > np.percentile(af568, 50) #useless just an initialisation
        for i in range(len(empty_space_568)):
            try:
                empty_space_568[i] = af568[i] > np.percentile(af568[i][img_dapi_mask[i] > 0], threshold_percent)
            except Exception as e: # in case there is no cell in one slide
                print(e)
        empty_space_647 = af647 > np.percentile(af647, 50)
        for i in range(len(empty_space_647)):
            try:
                empty_space_647[i] = af647[i] > np.percentile(af647[i][img_dapi_mask[i] > 0], threshold_percent)
            except Exception as e:
                print(e)
        empty_space = np.zeros(shape=empty_space_647.shape)
        for i in range(len(empty_space)):
            empty_space[i] = np.logical_and(empty_space_568[i] > 0, empty_space_647[i] > 0).astype(int) # the none empty space has to be present in both smfish channel
    elif heuristic == "af568_only":
        empty_space = af568 > np.percentile(af568, threshold_percent)
        for i in range(len(empty_space)):
            empty_space[i] = af568[i] > np.percentile(af568[i], threshold_percent)
    elif heuristic == "no_empty_space_removing":
        empty_space = np.ones(af568.shape)
    else:
        raise ValueError('%s is not an implemented heurisitc '  % str(heuristic))

    inverted_mask = np.ones(img_dapi_mask.shape) - (img_dapi_mask != 0).astype(np.int)
    if len(img_dapi_mask.shape) == 3:
        distance = ndi.distance_transform_edt(inverted_mask,
                                              sampling=[300, 103, 103])
    else:
        distance = ndi.distance_transform_edt(inverted_mask)  # compute distance map to border
    labels_with_empty = watershed(image=distance, markers=img_dapi_mask, mask=empty_space)
    return labels_with_empty


def get_adjacent_list(img_dapi_mask, labels_with_empty):
    """
   Parameters
    ----------
    img_dapi_mask : numpy
        3D mask
    labels_with_empty : numpy array
     voronoide diagram without the empty space

    Returns
    -------
    adjacent_list : TYPE
     adjacencet list of the graph
    dico_ngb : TYPE
        dictionary of neigbors

    """
    time.time()
    nuclei_list = np.unique(img_dapi_mask)[1:]  # remove  backgroud
    dico_ngb = {}
    for nucleus_pos in nuclei_list:
        t = time.time()
        tess_curent_nuc = np.max((img_dapi_mask == nucleus_pos).astype(int) * labels_with_empty)
        if tess_curent_nuc == 0: #the cell is in an empty space, does it means the empty threshold should be change ?
            dico_ngb[nucleus_pos] = []
            print("In empty space  %s" % str(nucleus_pos))
        else:
            frontiers = ndi.maximum_filter((labels_with_empty == tess_curent_nuc).astype(int), size=3)
            neighbors_tess = np.unique(frontiers * labels_with_empty)[1:]  # remove the zero of the background
            dico_ngb[nucleus_pos] = list(set(neighbors_tess) - set([nucleus_pos])) # remove self loop
        print(time.time() - t)

    adjacent_list = []

    for k in dico_ngb.keys():
        for node in dico_ngb[k]:
            adjacent_list.append((k, node))
    return adjacent_list, dico_ngb

def get_dico_centroid(img_dapi_mask):
    """

    Parameters
    ----------
    img_dapi_mask

    Returns
    dico key centroid index , value centroid
    -------

    """
    nuclei_list = np.unique(img_dapi_mask)
    dico_centroid = {}
    for nucleus_pos in nuclei_list:
        dico_centroid[nucleus_pos] = np.mean(np.nonzero((img_dapi_mask == nucleus_pos)), axis=1).astype(int)
    return dico_centroid


def remove_long_edge(img_dapi_mask, adjacent_list, dico_centroid, threshold=30900):
    """

    Parameters
    ----------
    img_dapi_mask
    adjacent_list
    dico_centroid
    threshold

    Returns
    return a new adjacent_list without the longest edge

    """
    new_adjacent_list = []
    for edge in adjacent_list:
        if edge[0] in set(dico_centroid.keys()) and edge[1] in set(dico_centroid.keys()):
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            if point1.ndim == 2:
                point1 = point1 * np.array([103, 103])
                point2 = point2 * np.array([103, 103])
            else:
                point1 = point1 * np.array([300, 103, 103])
                point2 = point2 * np.array([300, 103, 103])

            if np.linalg.norm(point1 - point2) < threshold:
                new_adjacent_list.append(edge)
    return new_adjacent_list



def remove_self_loop_dico(dico_nb):
    """

    Parameters
    ----------
    dico_nb

    Returns
    -------

    """
    res = {}
    for k in dico_nb:
        if k != 0: #remove background
            res[k] = list(set(dico_nb[k])- set([k]))
    return res

def remove_self_loop_adlist(ad_list):
    """

    Parameters
    ----------
    ad_list

    Returns
    -------

    """
    res = []
    for edge in ad_list:
        if edge[0] != edge[1] and edge[0] != 0 and edge[1] != 0:
            res.append(edge)
    return res

def dico_to_ad_list(dico):
    ad_list = []
    for k in dico.keys():
        for ngb in dico[k]:
            ad_list.append((k, ngb))
    return ad_list


def ad_list_to_dico(ad_list, add_lonely_node  = []):
    ad_dico = {}
    nodes = []
    for t in ad_list:
        nodes += [t[0], t[1]]
    for nn in nodes + add_lonely_node:
        ad_dico[int(nn)] = []
    for t in ad_list:
        ad_dico[t[0]].append(int(t[1]))
        ad_dico[t[1]].append(int(t[0]))
    for k in ad_dico:
        ad_dico[k] = list(set(ad_dico[k]))
    return ad_dico

def get_weighted_networkx_graph(adjacent_list, node_list=None,  use_weight =False, dico_centroid=None, img_dapi_mask = None):
    """

    Parameters
    ----------
    adjacent_list: list, ad of the graph
    use_weight: Bool, if false return an weithed networkx graph
    dico_centroid: dictionary key =
    img_dapi_mask

    Returns
    -------

    """


    G = nx.Graph(tuple(map(tuple, adjacent_list)))
    if node_list is not None:
        G.add_nodes_from(node_list)  #add solitary node

    if use_weight and dico_centroid is None:
        dico_centroid = get_dico_centroid(img_dapi_mask)
    for edge in adjacent_list:

        if use_weight:
            point1 = dico_centroid[edge[0]] * np.array([3, 1, 1])
            point2 = dico_centroid[edge[1]] * np.array([3, 1, 1])
            G[edge[0]][edge[1]]['weight'] = np.linalg.norm(point1 - point2)
        else:
            G[edge[0]][edge[1]]['weight'] = 1
    return G