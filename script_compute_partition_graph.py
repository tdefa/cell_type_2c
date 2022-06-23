
#%%
from graph.toolbox import get_dico_centroid, get_louvain_partition, compute_centrality_harmonic, compute_centrality_betweeness
import time
import community
import argparse
import time
import os
from os import listdir
from os.path import isfile, join

import tifffile
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality, betweenness_centrality, edge_betweenness_centrality

from graph.toolbox import remove_self_loop_adlist

from spots.post_processing import erase_solitary
import numpy as np

#if __name__ == "__main__":
folders_list = ["210426_repeat3/",  # ok
                      "200828-NIvsIR5M/00_Capillary_EC/",
                      "200828-NIvsIR5M/00_Large_Vessels/",  # okd
                    "200828-NIvsIR5M/00_Macrophages/",  # okd
                     "200908_CEC/",  # okd

                    "201030_fridyay/",  # ok
                    "210205_Prolicence/aCap_prolif/",  # okd
                    "210205_Prolicence/aCap_senes/",  # okd
                    "210219_myo_fibros_y_macrophages/",  # okd
                    "210412_repeat_fibro/IR5M/",  # okd
                    "210412_repeat_fibro/NI/",  # okd
                    "210413_rep2/",  # okd
                    "210425_angiogenesis/",  # ok
                    "200908_fibrosis/",  # ok
                    "201127_AM_fibro/",  # okd
                    "210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/"
                    ]
for path_to_folder in folders_list[14:]:
    parser = argparse.ArgumentParser(description='generate_graph')
    parser.add_argument('-ptt',
                    "--path_to_take",
                    type=str,
                    default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
                    help='path_to_take')

    parser.add_argument('-ptf', "--path_to_folder", type=str,
                    default=path_to_folder,
                    help='')

    parser.add_argument("--name_save", type=str,
                        default = "dd_t5_2907",
                        help = '')
    parser.add_argument("--port", default=39949)
    parser.add_argument("--mode", default='client')
    args = parser.parse_args()
    print(args)
    folder = args.path_to_take + args.path_to_folder
    path_to_ad = folder + "adjacent_list/"

    path_output_segmentaton = folder + "tiff_data/predicted_mask_dapi/"
    path_dapi = folder + "tiff_data/predicted_mask_dapi/"
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
         isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
    dico_partition = {}
    for f in onlyfiles:
            print(f)
            t = time.time()
            ad_list = remove_self_loop_adlist(np.load(path_to_ad + "ad_list" +args.name_save + f + ".npy"))
            img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + f)
            img_dapi_mask = erase_solitary(img_dapi_mask)
            print(time.time()-t)
            dico_centroid = get_dico_centroid(img_dapi_mask)
            print(time.time() - t)
            partition = get_louvain_partition(ad_list, dico_centroid)
            print(time.time() - t)
            path_dapi = folder + "tiff_data/dapi/"
            dapi = tifffile.imread(path_dapi + "dapi_" + f)

            modularity = community.modularity(partition[0], partition[1])
            print("modularity %s" % modularity)
            G = nx.Graph(ad_list)
            dico_betweenness_centrality = betweenness_centrality(G, weight='weight')
            dico_edge_betweenness = edge_betweenness_centrality(G, weight='weight')
            dico_harmonic_centrality = harmonic_centrality(G, distance='weight')

            dico_partition[f] = [("partition", partition), ("modularity", modularity),
                                 ("dico_betweenness_centrality", dico_betweenness_centrality), ("dico_edge_betweenness", dico_edge_betweenness),
                                 ('dico_harmonic_centrality', dico_harmonic_centrality)]
    np.save(folder + "dico_partition2408", dico_partition)