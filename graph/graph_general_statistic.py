
#######
# graph statistic
########
#%%

import time
import numpy as np
import community
import networkx as nx

from scipy import ndimage as ndi

from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality, harmonic_centrality


def get_neighbors_empty(labels_with_empty, max_filter_vacumm = 7, max_filter_frontier = 8):
    """

    Parameters
    ----------
    labels_with_empty: lacunar mask
    max_filter_vacumm: max filter kernel to get compact non-lacunar area
    max_filter_frontier: max filter kernel to get compact lacunar area

    Returns
    -------

    """
    labels_with_empty_filter = ndi.maximum_filter(labels_with_empty,
                                                  size=max_filter_vacumm)
    frontiers = ndi.maximum_filter((labels_with_empty_filter == 0).astype(int),
                                   size=max_filter_frontier)
    contact_wt_empty = np.unique(frontiers * labels_with_empty)
    return contact_wt_empty





def compute_centrality_betweeness(adjacent_list, dico_centroid):
    """
    Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v:
    Parameters
    ----------
    adjacent_list
    dico_centroid

    Returns
    list of (node_id , centrality) sorted by centrality

    """
    G = nx.Graph(adjacent_list)
    if dico_centroid is not None:
        for edge in adjacent_list:
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            G[edge[0]][edge[1]]['weight'] = np.linalg.norm(point1 - point2)
            G[edge[1]][edge[0]]['weight'] = np.linalg.norm(point1 - point2)

    dico_centrality = betweenness_centrality(G, weight='weight')
    kv = [(k, dico_centrality[k]) for k in dico_centrality]
    kv = sorted(kv, key=lambda tup: tup[1])
    return kv, dico_centrality


def compute_edge_centrality_betweeness(adjacent_list, dico_centroid):
    """
    Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v:
    Parameters
    ----------
    adjacent_list
    dico_centroid

    Returns
    list of (node_id , centrality) sorted by centrality

    """
    G = nx.Graph(adjacent_list)
    if dico_centroid is not None:
        for edge in adjacent_list:
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            G[edge[0]][edge[1]]['weight'] = np.linalg.norm(point1 - point2)
            G[edge[1]][edge[0]]['weight'] = np.linalg.norm(point1 - point2)

    dico_centrality = edge_betweenness_centrality(G, weight='weight')
    kv = [(k, dico_centrality[k]) for k in dico_centrality]
    kv = sorted(kv, key=lambda tup: tup[1])
    return kv, dico_centrality



def compute_centrality_harmonic(adjacent_list, dico_centroid):
    """

    Parameters
    ----------
    adjacent_list
    dico_centroid

    Returns
    -------

    """
    G = nx.Graph(adjacent_list)
    if dico_centroid is not None:
        for edge in adjacent_list:
            point1 = dico_centroid[edge[0]]
            point2 = dico_centroid[edge[1]]
            G[edge[0]][edge[1]]['weight'] = np.linalg.norm(point1 - point2)
            G[edge[1]][edge[0]]['weight'] = np.linalg.norm(point1 - point2)

    dico_centrality = harmonic_centrality(G, distance='weight')
    kv = [(k, dico_centrality[k]) for k in dico_centrality]
    kv = sorted(kv, key=lambda tup: tup[1])
    return kv, dico_centrality

def compute_nb_edge(dico_nb, positive_af568, positive_af647):
    """
    Parameters
    ----------
    dico_nb : a dictionary of neighbors : key a node, value its neighbors
    positive_af568: list of positive node/nuclei to af568
    positive_af647: list of positive node/nuclei to af647

    Returns
    The number of edge af568-af568 ect
    list of the degree of the node positvie to xxx
    -------

    """
    nb_af568 = 0
    nb_af647 = 0
    nb_both = 0
    degre_list_af568  = []
    degre_list_af647 = []
    for k in dico_nb: #dico_nb is a dic
        if k in positive_af568:
            nb_af568 += len(set(positive_af568).intersection(set(dico_nb[k])))
            nb_both += len(set(positive_af647).intersection(set(dico_nb[k])))
            degre_list_af568.append(len(dico_nb[k]))

        if k in positive_af647:
            nb_af647 += len(set(positive_af647).intersection(set(dico_nb[k])))
            nb_both += len(set(positive_af568).intersection(set(dico_nb[k])))
            degre_list_af647.append(len(dico_nb[k]))

    nb_af568 = nb_af568/2
    nb_af647 = nb_af647/2
    nb_both = nb_both/2
    return nb_af568,nb_af647, nb_both, degre_list_af568, degre_list_af647






def compute_nb_empty_contact(dico_nb_with_zero, positive_af568 = None, positive_af647 = None):
    """
    Parameters
    ----------
    dico_nb_with_zero : a dictionary of neighbors : key a node, value its neighbors (with 0 background edge)
    positive_af568: list of positive node  /  nuclei to af568
    positive_af647: list of positive node / nuclei to af647

    Returns
    The number of edge af568-af568 af568-af647 if dico_zero is about af568
    list of the degree of the node positive to xxx
    -------
    """
    if positive_af568 is None and positive_af647 is None:
        nb_zero = 0
        for k in dico_nb_with_zero: #dico_nb is a dic
            if 0 in dico_nb_with_zero[k]:
                nb_zero += 1
        proportion =  nb_zero / len(dico_nb_with_zero.keys())
        return proportion, nb_zero
    else:
        nb_af568 = 0
        nb_af647 = 0
        for k in dico_nb_with_zero: #dico_nb is a dic
            if 0 in dico_nb_with_zero[k]:
                if k in positive_af568:
                    nb_af568 += 1
                if k in positive_af647:
                    nb_af647 += 1
        per_af568 = nb_af568 / len(positive_af568) if len(positive_af568) > 0 else None
        per_af647 = nb_af647 / len(positive_af647) if len(positive_af647) > 0 else None
    return per_af568, per_af647, nb_af568, nb_af647


def simmulation_maslov_sneppen(g, positive_node, nb_simulation, niter = 1000, second_positive_node = None):
    """
    use a nul model based an random rewiring described in :
    Specificity and stability in topology of protein networks Sergei Maslov  1 , Kim Sneppen
    Parameters
    ----------
    g
    positive_node
    nb_simulation
    niter
    second_positive_node

    Returns
    -------

    """
    list_nb_contact = []
    t = time.time()

    for sim in range(nb_simulation):
        gr = nx.algorithms.smallworld.random_reference(g, niter=niter,
                                                  connectivity=False,
                                                  seed=None)
        if second_positive_node is None:
            list_nb_contact.append(len(gr.subgraph(positive_node).edges))
        else:
            list_nb_contact.append(len(gr.subgraph(positive_node).edges))

    print(time.time() - t)
    return np.mean(list_nb_contact), np.var(list_nb_contact)

#%%
def simmulation_random_node(g, positive_node,  nb_simulation = 10000, second_positive_node = None, exclusive = True):
    """
    Its simply changes the position of the node in a random order
    Parameters
    ----------
    g
    positive_node
    nb_simulation
    niter
    second_positive_node
    exclusive: make the intersction between second_positive_node and positive_node empty. If exclusive is false the result is not symmetric
    Returns
    -------

    """
    list_nb_contact = []
    t = time.time()
    list_of_nodes = list(g.nodes)
    if second_positive_node is not None:
        if exclusive:
            nb_first_node = len(set(positive_node)- set(second_positive_node) )
            nb_second_node = len(set(second_positive_node) - set(positive_node))
        else:
            nb_first_node = len(set(positive_node))
            nb_second_node = len(set(second_positive_node))
    for sim in range(nb_simulation):
        if second_positive_node is None:
            random_positive_node = np.random.choice(list_of_nodes, size=len(set(positive_node)), replace=False, p=None)
            list_nb_contact.append(len(g.subgraph(random_positive_node).edges))
        else:
            random_positive_first_node = np.random.choice(list_of_nodes, size=nb_first_node, replace=False, p=None)
            random_positive_second_node = np.random.choice(list_of_nodes, size=nb_second_node, replace=False, p=None)
            partition = {}
            for n in list_of_nodes:##construct the partition key 0 negative, 1 positve to 1, 2 positive to 2
                if n in random_positive_first_node:
                    partition[n] = 1
                elif n in random_positive_second_node:
                    partition[n] = 2
                else:
                    partition[n] = 0
            induced_graph  = community.induced_graph(partition, g, weight='weight')
            #print(induced_graph.edges(data=True))
            list_nb_contact.append(list(induced_graph.edges(data=True))[-1][-1]['weight'])

    #print(time.time() - t)
    return np.mean(list_nb_contact), np.var(list_nb_contact)