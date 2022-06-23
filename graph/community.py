

#%%
import time


import numpy as np
import community
import networkx as nx



from graph.graph_construction import remove_self_loop_dico



def get_louvain_partition(adjacent_list, dico_centroid=None, use_weight = True): #to check the egde weight
    """

    Parameters
    ----------
    adjacent_list
    dico_centroid
    use_weight

    Returns
    partition: dico key=index node, value = community
    -------

    """
    G = nx.Graph(tuple(map(tuple, adjacent_list)))
    if dico_centroid is not None:
        for edge in adjacent_list:
            point1 = dico_centroid[edge[0]] * np.array([3, 1, 1])
            point2 = dico_centroid[edge[1]] * np.array([3, 1, 1])
            if use_weight:
                G[edge[0]][edge[1]]['weight'] = np.linalg.norm(point1 - point2)
            else:
                G[edge[0]][edge[1]]['weight'] = 1
    partition = community.best_partition(G, weight='weight', random_state=2)
    return partition, G


def compute_community_distance(dico_part , dico_ad):
    """
    https://arxiv.org/pdf/1509.08295.pdf  ' simple indicator inspired by the well-known Rao-Stirling index [17, 27–29], as this indicator is known to quantify
the ability of nodes to connect different communities.'
    Parameters
    ----------
    dico_part: dico, keys nodes values community index
    dico_ad : dico, keys neighbours, values community index

    Returns
    dico_rao_stirling key node index, value
    -------

    """
    dico_ad_c = remove_self_loop_dico(dico_ad)
    dico_community_distance = {}
    community_index = np.unique(list(dico_part.values()))
    #initialisation
    for c_i in range(len(community_index)):
        for c_i2 in range(c_i):
            if c_i2 != c_i:
                dico_community_distance[(c_i2, c_i)] = 0
    # compute distance between communitu
        for node_pp in dico_ad_c.keys():
            try:
                c_node = dico_part[node_pp]
                for neighb in dico_ad_c[node_pp]:
                    if dico_part[neighb] < c_node:
                        dico_community_distance[(dico_part[neighb], c_node)] += 1
                    if c_node < dico_part[neighb]:
                        dico_community_distance[(c_node, dico_part[neighb])] += 1
            except Exception as e:
                print(repr(e))
        for node_pp in dico_community_distance.keys():
            dico_community_distance[node_pp] = 1/ dico_community_distance[node_pp] if dico_community_distance[node_pp] > 0 else np.inf

        # compute the Rao_Stirling index for graph
        dico_rao_stirling = {}
        for node_pp in dico_ad_c.keys():
            try:
                dico_rao_stirling[node_pp] = [0 for i in range(len(community_index))]
                c_node = dico_part[node_pp]
                for neighb in dico_ad_c[node_pp]:
                    if dico_part[neighb] < c_node:
                        dico_rao_stirling[node_pp][dico_part[neighb]] = dico_community_distance[(dico_part[neighb], c_node)]
                    if c_node < dico_part[neighb]:
                        dico_rao_stirling[node_pp][dico_part[neighb]] = dico_community_distance[(c_node, dico_part[neighb])]
            except Exception as e:
                print(repr(e))
        for node_pp in dico_rao_stirling.keys():
            dico_rao_stirling[node_pp] = np.sum(dico_rao_stirling[node_pp])

    return dico_rao_stirling

def nb_cell_type_per_community(partition_dico, dico_type, f):
    """

    Parameters
    ----------
    partition: community partition of a sample.
    positive_af568: index of node positive to af568
    positive_af647: index of node positive to af647

    Returns
    dico key community index, value [nb of cell  in the community, nb AF568+, nb AF647+, clustering coefficient]
    -------

    """

    positive_568 = [dico_type[f][5][i][3] for i in range(len(dico_type[f][5]))]
    print("af568 %s" % len(positive_568))
    positive_647 = [dico_type[f][6][i][3] for i in range(len(dico_type[f][6]))]

    print("af647 %s"  % len(positive_647))
    partition = partition_dico[f][0][1][0]
    g = partition_dico[f][0][1][1] #be carreful to check that there is no self loop (script_compute_partition_graph.py)
    dico_clustering = nx.clustering(g, weight='weight')
    print(dico_clustering)
    ### nb partition
    nb_partition = np.unique(list(partition.values()))
    dico_com_features = {}

    for part in nb_partition:
        dico_com_features[part] = np.array([ 0, 0,  0,  0], dtype=float) #[nb af568, nb af647, total_nb_cell, clustering coeff]

    ### count the nb of cell type per comunity

    for node_index in partition.keys():
        local_part = partition[node_index]
        dico_com_features[local_part][2] += 1
       # print("dico cl %s " %  dico_clustering[node_index])
        dico_com_features[local_part][3]  += dico_clustering[node_index]
        if node_index in positive_568:
            dico_com_features[local_part][0] += 1
        if node_index in positive_647:
            dico_com_features[local_part][1] += 1

    for part in nb_partition:
        #print(dico_com_features[part][3])
        dico_com_features[part][3] = dico_com_features[part][3] / dico_com_features[local_part][2]
    return dico_com_features


def compute_clustering_coeff(dico_partition, file):
    """

    Parameters
    ----------
    dico_partition key files values dict: (key=node,values=commu)
    file:

    Returns
    -------

    """
    partition = dico_partition[file][0][1][0]
    graph = dico_partition[file][0][1][1]
    list_of_commu = np.unique(list(partition.values()))
    dico_commu = {}
    for commu in list_of_commu:
        dico_commu[commu] = []
        for node in partition:
            if partition[node] == commu:
                dico_commu[commu].append(node)
    dico_commu_clustering = {}
    for commu in dico_commu.keys():
        avg_per_commu = nx.algorithms.cluster.average_clustering(graph, np.array(dico_commu[commu]))
        avg_all_graph = nx.algorithms.cluster.average_clustering(graph)
        dico_commu_clustering[commu] = avg_per_commu,
    return  [dico_commu_clustering, avg_all_graph, nx.algorithms.cluster.clustering(graph)]
## compute clustering coefficient of graph


#%%

if __name__ == "__main__":

    path_to_take = "/media/tom/Elements/to_take/"
    list_folder = [
        "200828-NIvsIR5M/00_Capillary_EC/",  # ok spot ok image, on image is wrong
        "200828-NIvsIR5M/00_Large_Vessels/",  # pb to rerun
        "200828-NIvsIR5M/00_Macrophages/",  # ok spot
        "200908_CEC/",
        "200908_fibrosis/",
        "201030_fridyay/",
        "201127_AM_fibro/",  ##pb
        "210205_Prolicence/aCap_prolif/",
        "210205_Prolicence/aCap_senes/",
        "210219_myo_fibros_y_macrophages/",
        "210412_repeat_fibro/IR5M/",
        "210412_repeat_fibro/NI/",
        "210413_rep2/",
        "210425_angiogenesis/",
        "210426_repeat3/",
        "210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/"
    ]


    dico_partion_name = "dico_partition2408.npy"


    for folder in list_folder:
        dico_clustering_coef = {"readme": "dico_commu_clustering, avg_all_graph, nx.algorithms.cluster.clustering(graph)"}
        dico_partition = np.load(path_to_take + folder + dico_partion_name, allow_pickle=True).item()
        for file in dico_partition:
            dico_clustering_coef[file] = compute_clustering_coeff(dico_partition, file)
        np.save(path_to_take + folder + "dico_clustering_coef",dico_clustering_coef)









