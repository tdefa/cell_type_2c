
#%% file using toolbox function on my folder logic (comment je l'ai rangé)


import argparse
import tifffile
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from graph.graph_construction import remove_self_loop_dico, remove_self_loop_adlist, get_weighted_networkx_graph
from graph.graph_general_statistic import  compute_nb_edge, compute_nb_empty_contact , simmulation_maslov_sneppen, simmulation_random_node
from graph.community import compute_community_distance, nb_cell_type_per_community
from spots.post_processing import erase_solitary
from os import listdir
from os.path import isfile, join
import os
from utils import get_dye, get_mouse_name
import time
#%%
def script_compute_general(folder,
                script_processing_fct,
                name_ad = "dd_t5_2907",
                name_dico_type = "dico_stat_2106.npy",
                name_save = "dico_empty_neighbour"):

    """
    compute the statistic define by processing_fct for each folder
    Parameters
    ----------
    folder : path to the folder of data
    script_processing_fct: function that take in arg the dico of type,
    the  path to folder, and name_ad and return the desired stat: example the mean node degree for cell positive to CY3
    name_ad  :name of dico of graph to laod
    name_dico_type : name of dico type
    name_save:

    Returns
    the dictionarry key: name file, value: statisic define by script_processing_fct
    -------

    """

    path_output_segmentaton = folder + "tiff_data/predicted_mask_dapi/"
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
                 isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

    dico_type = np.load(folder + name_dico_type,
                        allow_pickle=True).item()

    dico = {}
    i_f = 0
    for f in onlyfiles:
        print(folder+str(i_f))
        print(f)
        i_f += 1
        if f in list(dico.keys()):
            continue
        if f in ["09_NI_Ptprb-Cy3_Mki67-Cy5_02.tiff", "06_IR5M_Lamp3-Cy3_Serpine1-Cy5_011.tiff"]:
            dico[f] = ["type no available ?"]
        else:
            value = script_processing_fct(dico_type, folder, name_ad, f)
            dico[f] = value
        np.save(folder + name_save, dico)
    return dico
#%%
def compute_gene(  ### FORMER compute_general
    gene,
    folders_list,
    compute_gene_fct,
    path_to_take = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
    dico_name = "dico_degree_volume",
    folder_save ="degree_volume" ,
    dico_stat_name = "dico_stat_2106"):
    """
    use the statistics computed by script_compute_general and aggregate them for each gene
    Parameters
    ----------
    gene :   ex ['Lamp3']
    folders_list:  ex     folders_list = ["210426_repeat3/",  "200828-NIvsIR5M/00_Capillary_EC/"]
    compute_gene_fct: aggregate/ custom the value from the dictonarry load from dico_name using as input
                     dico, dye_type, key_cell_name, dico_type
    path_to_take: path to the folder of folder to aggregate
    dico_name: dico save by script_compute_general
    folder_save: name of the result to save
    dico_stat_name :  cell type claasif dico to load

    Returns
    -------dico_NI_ratio, dico_IR5M_ratio where there are the deisired statisic for each sample with the targeted gene

    """
    dico_NI_ratio = {}
    dico_IR5M_ratio = {}
    for folder_name in folders_list:
        ## load  dictionary from script_compute_general
        dico = np.load(path_to_take + folder_name + "dico_" + dico_name + ".npy", allow_pickle=True).item()

        ## load the dictionary from main_cluster it is sometime usefull
        dico_type = np.load(path_to_take + folder_name + dico_stat_name + ".npy", allow_pickle=True).item()
        sorted_name = np.sort(list(dico.keys()))
        for key_cell_name in sorted_name:
            if key_cell_name in ["09_NI_Ptprb-Cy3_Mki67-Cy5_02.tiff", "06_IR5M_Lamp3-Cy3_Serpine1-Cy5_011.tiff"]:
                continue
            ### implementation double gene
            if len(gene) >1 and ((gene[0] == ['Serpine1'] or  gene[0] == ['Mki67']) and (gene[1] != ['Serpine1'] or  gene[1] != ['Mki67']))  :
                dye_type ="cell_state_type"
                if not any(word in key_cell_name for word in gene[0]):
                    continue
                if not any(word in key_cell_name for word in gene[1]):
                    continue
            else:
                if not any(word in key_cell_name for word in gene):
                    continue
                dye_type = get_dye(gene, key_cell_name) # I remove the safeguard for the  ["type no available ?"] be carefull

            value = compute_gene_fct(dico, dye_type, key_cell_name, dico_type)
            if value is None:
                continue
            if  any(word in key_cell_name for word in ["NI", "Ctrl"]):
                #if "1225" in key_cell_name:
                #    print("NI %s" % key_cell_name)
                #if "1230" in key_cell_name:
                #    print("NI %s" % key_cell_name)
                dico_NI_ratio[key_cell_name] = value
            if  any(word in key_cell_name for word in ["IR5M"]):
                dico_IR5M_ratio[key_cell_name] = value

    #now save the dico :)
    if not os.path.exists(path_to_take + folder_save +"/"):
        os.mkdir(path_to_take + folder_save + "/")
    if dye_type == "cell_state_type":
        np.save(path_to_take + folder_save + "/" + gene[0][0]+"_" + gene[1][0] + "NI", dico_NI_ratio)
        np.save(path_to_take + folder_save + "/" + gene[0][0]+"_" + gene[1][0] + "IR5M", dico_IR5M_ratio)
    else:
        print(path_to_take + folder_save + "/" + gene[0] + "NI")
        np.save(path_to_take + folder_save + "/" + gene[0] + "NI", dico_NI_ratio)
        np.save(path_to_take + folder_save + "/" + gene[0] + "IR5M", dico_IR5M_ratio)
    return dico_NI_ratio, dico_IR5M_ratio


########
# script function for script_compute_general processing_fct
########
#%%

def script_processing_fct_expected_edge(dico_type, folder, name_ad, f, null_model = "maslov_sneppen", cell_pair = False):
    """
    Function that compute the number of edge af568-af568 af647-af647, af647-af568, nb total
    Parameters
    ----------
    null_model: choose in "configuration", "Maslov_Sneppen", "node_permutation"
    dico_type
    folder
    name_ad
    f

    Returns
    -------

    """
    assert null_model in ["configuration", "maslov_sneppen", "node_permutation"]
    if f in ["09_NI_Ptprb-Cy3_Mki67-Cy5_02.tiff", "06_IR5M_Lamp3-Cy3_Serpine1-Cy5_011.tiff"]:
            value = ["type no available ?"]
            return value
    #try:
    t = time.time()
    print(f)
    print(null_model)
    ad_list = np.load(folder  + "adjacent_list/" + "ad_list" + name_ad + f + ".npy")

    ad_list = remove_self_loop_adlist(ad_list)
    dico_nb = remove_self_loop_dico(np.load(folder + "adjacent_list/"+ "dico_no" + name_ad + f + ".npy",
                                            allow_pickle=True).item())
    node_list = list(dico_nb.keys())
    g = get_weighted_networkx_graph(ad_list, node_list = node_list, use_weight=False, dico_centroid=None, img_dapi_mask=None)

    nb_total = len(ad_list) / 2 # because the edges in the adjacentlist are coUnted as directed

    positive_af568 = [dico_type[f][5][i][3] for i in range(len(dico_type[f][5]))]
    positive_af647 = [dico_type[f][6][i][3] for i in range(len(dico_type[f][6]))]
    nb_af568, nb_af647, nb_both, degre_list_af568, degre_list_af647= compute_nb_edge(dico_nb,
                                                                                      positive_af568,
                                                                                      positive_af647)
    if null_model =="configuration":
        expected_af568 = 0.5 * np.sum([[degre_list_af568[i] * degre_list_af568[j] / (2 *  nb_total) for i in range(len(degre_list_af568))] for j in range(len(degre_list_af568))])
        expected_af647 = 0.5 * np.sum([[degre_list_af647[i] * degre_list_af647[j] / (2 *  nb_total) for i in range(len(degre_list_af647))] for j in range(len(degre_list_af647))])
        expected_af568_af647 = None
        expected_af568_af647_pure =None
    if null_model == "maslov_sneppen":
        expected_af568 = simmulation_maslov_sneppen(g, positive_af568, nb_simulation = 500, niter = 4)
        expected_af647 = simmulation_maslov_sneppen(g, positive_af647, nb_simulation = 500, niter = 4)
        expected_af568_af647_pure= None
        expected_af568_af647 = None

    if null_model == "node_permutation":
        expected_af568 = simmulation_random_node(g, positive_af568, nb_simulation=10000, second_positive_node=None)
        expected_af647 = simmulation_random_node(g, positive_af647, nb_simulation=10000, second_positive_node=None)
        expected_af568_af647_pure = simmulation_random_node(g, positive_af568, nb_simulation=10000,
                                                            second_positive_node=positive_af647, exclusive = True)
        expected_af568_af647 = None
    print(time.time() -t)




    value = [expected_af568 , expected_af647, expected_af568_af647_pure,  expected_af568_af647,
             ("af568", nb_af568), ("af647", nb_af647),
             ("both", nb_both), ("total", nb_total),
            degre_list_af568, degre_list_af647]
    #except Exception as e:
    #        print(e)
    #        #assert f in["09_NI_Ptprb-Cy3_Mki67-Cy5_02.tiff", "06_IR5M_Lamp3-Cy3_Serpine1-Cy5_011.tiff"]
    #        value = ["type no available ?", str(e)]
    return value

#%%

def script_processing_fct_degree(dico_type, folder, name_ad, f):
    """
    Parameters see script_compute_general
    ----------
    dico_type
    folder
    name_ad
    f
    Returns
    -------

    """
    path_to_ad = folder + "adjacent_list/"
    dico_ad = remove_self_loop_dico(np.load(path_to_ad + "dico_no" + name_ad + f + ".npy", allow_pickle=True).item())

    index_af568 = [c[3] for c in dico_type[f][5]]
    index_af647 = [c[3] for c in dico_type[f][6]]
    index_af568 = set(index_af568)
    index_af647 = set(index_af647)

    list_node_degree_all = [len(dico_ad[k]) for k in dico_ad]
    list_degree_af568 = [len(dico_ad[k]) for k in index_af568]
    list_degree_af647 = [len(dico_ad[k]) for k in index_af647]
    return [list_node_degree_all, list_degree_af568, list_degree_af647]



def script_processing_fct_empty(dico_type, folder, name_ad, f):
    path_to_ad = folder + "adjacent_list/"
    nb_total = dico_type[f][0]
    index_af568 = [c[3] for c in dico_type[f][5]]
    index_af647 = [c[3] for c in dico_type[f][6]]
    index_af568 = list(set(index_af568))
    index_af647 = list(set(index_af647))
    array = np.load(path_to_ad + "dico_" + name_ad + f + ".npy", allow_pickle=True)

    return [nb_total, index_af568, index_af647, array]

def script_processing_fct_rao_stirling(dico_type, folder, name_ad, f):
    dico_partition = np.load(folder  + "dico_partition2408.npy", allow_pickle = True).item()
    path_to_ad = folder + "adjacent_list/"
    dico_ad = remove_self_loop_dico(np.load(path_to_ad + "dico_no" + name_ad + f + ".npy", allow_pickle=True).item())
    dico_rao_stirling = compute_community_distance(dico_partition[f][0][1][0], dico_ad)
    return dico_rao_stirling

def script_processing_fct_community_features(dico_type, folder, name_ad, f):
    dico_partition = np.load(folder  + "dico_partition2408.npy", allow_pickle = True).item()
    path_to_ad = folder + "adjacent_list/"
    dico_com_features = nb_cell_type_per_community(dico_partition, dico_type, f)
    return dico_com_features




##########
# function as input for compute_gene
########
#%%

def compute_gene_fct_compute_expected_edge(dico, dye_type, key_cell_name, dico_type = None):
    """
    Parameters
    ----------
    dico
    dye_type
    key_cell_name
    dico_type
    Returns
    -------
    """
    if ['type no available ?'] == dico[key_cell_name]:
        print(key_cell_name)
        return None
    [expected_af568, expected_af647, expected_af568_af647_pure, expected_af568_af647, nb_af568,  nb_af647, nb_both,  nb_total,  degre_list_af568, degre_list_af647] = dico[key_cell_name]
    if type(expected_af568) == type(()) or type(expected_af568) ==type([]):
        expected_af568 = expected_af568[0]
        expected_af647 = expected_af647[0] #pas tres bien codé
        print(nb_both)
        print(expected_af568_af647_pure)
    if dye_type == "Cy3":
        # af568
        if len(degre_list_af568) == 0:
            return None
        value = [expected_af568, nb_af568, nb_af568[1]/expected_af568]
    elif dye_type == "Cy5":
        # af647
        if len(degre_list_af647) == 0:
            return None
        print(expected_af647)
        value = [expected_af647, nb_af647, nb_af647[1]/expected_af647]
    elif dye_type == "cell_state_type":
        if type(expected_af568) == type(()) or type(expected_af568) == type([]):

            expected_af568_af647_pure = expected_af568_af647_pure[0] if expected_af568_af647_pure[0] is not None else np.inf
        # af647
        if len(degre_list_af568) == 0 or len(degre_list_af647) == 0:
            return None
        value = [expected_af568_af647_pure, nb_both, nb_both[1]/expected_af568_af647_pure]
    print(value)
    return value
#%%
def compute_gene_fct_degree(dico, dye_type, key_cell_name, dico_type = None):
    if ['type no available ?'] == dico[key_cell_name]:
        return [None, None, None]
    [list_node_degree_all, list_degree_af568, list_degree_af647] = dico[key_cell_name]
    if dye_type == "Cy3":
        # af568
        value = [np.mean(list_degree_af568), list_degree_af568, list_node_degree_all]
    else:
        # af647
        value = [np.mean(list_degree_af647), list_degree_af647, list_node_degree_all]
    return value


def compute_gene_fct_compute_degree_volume(dico_empty, dye_type, key_cell_name):
    if dye_type == "Cy3":
        # af568
        value = dico_empty[key_cell_name][0]
    else:
        # af647
        value = dico_empty[key_cell_name][1]
    return value

def compute_gene_fct_compute_empty_neighbour(dico, dye_type, key_cell_name, dico_type = None):
    #[nb_total, index_af568, index_af647, array]
    # What I want return is (number of positive to index_af568 with no contact with air) / (total number of af568) {af647}
    nb_total, index_af568, index_af647, array = dico[key_cell_name]
    if dye_type == "Cy3":
        # af568
        value = len(set(index_af568) - set(array)) / len(index_af568) if len(index_af568) > 0 else None
    else:
        # af647
        value = len(set(index_af647) - set(array)) / len(index_af647) if len(index_af647) > 0 else None
    return value

def compute_gene_fct_mean_rao_sterling(dico_rs, dye_type, key_cell_name, dico_type = None):
    """
    Parameters
    ----------
    dico : dico of dico of rao_streling index dico_stat[f] = dico[no]
    dye_type
    key_cell_name
    dico_type   dico_stat[f] = [len(np.unique(img_dapi_mask))0, 1nb_no_rna, 2 nb_cy3,3 nb_cy5,4 nb_both,
                                  5  positive_cluster_568, 6 positive_cluster_647, 7 negative_cluster_568, 8 negative_cluster_647]
    Returns
    the mean value of the rao_sterling for the positve gene of this cell
    -------
    """
    dico_node_rs = dico_rs[key_cell_name]
    try:
        if dye_type == "Cy3":
            positive_cell = [dico_type[key_cell_name][5][i][3] for i in range(len(dico_type[key_cell_name][5]))]
        else:
            positive_cell = [dico_type[key_cell_name][6][i][3] for i in range(len(dico_type[key_cell_name][6]))]
            # af647
    except Exception as e:
        print()
        print(e)
        print(key_cell_name)
        print()
        return None

    #print(positive_cell)
    if len(positive_cell) > 0:
        rs_mean = []
        for node in positive_cell:
            rs_mean.append(dico_node_rs[node])
        value = [np.mean(rs_mean), np.var(rs_mean)]
        return value
    return None
#%%
def processing_fct_proportion_rao_stirling_index(dico_rs, dye_type, key_cell_name, dico_type = None,  threshold = 0.10):
    """
    Parameters
    ----------
    dico : dico of dico of rao_streling index dico_stat[f] = dico[no]
    dye_type
    key_cell_name
    dico_type   dico_stat[f] = [len(np.unique(img_dapi_mask))0, 1nb_no_rna, 2 nb_cy3,3 nb_cy5,4 nb_both,
                                  5  positive_cluster_568, 6 positive_cluster_647, 7 negative_cluster_568, 8 negative_cluster_647]
    Returns
    the mean value of the rao_sterling for the positve gene of this cell
    [Nb_connector of type / nb_total_connector, expected number of connector of type (proportion of cell  connected),
     list_connector value, binary list_positive]
    -------
    """
    dico_node_rs = dico_rs[key_cell_name]
    try:
        if dye_type == "Cy3":
            positive_cell = [dico_type[key_cell_name][5][i][3] for i in range(len(dico_type[key_cell_name][5]))]
        else:
            positive_cell = [dico_type[key_cell_name][6][i][3] for i in range(len(dico_type[key_cell_name][6]))]
        nb_total_nuclei = dico_type[key_cell_name][0]
    except Exception as e:
        print()
        print(e)
        print(key_cell_name)
        print()
        return None
    list_connector_value = []
    binary_list_positive = []
    list_connector_index_value = [k for k in dico_node_rs if dico_node_rs[k] > threshold]

    if len(positive_cell) > 0 and len(list_connector_index_value) > 0:
        nb_connector_total = len(list_connector_index_value)
        nb_connector_type = 0
        if nb_connector_total > 0:
            for node in list_connector_index_value:
                if node in positive_cell:
                    nb_connector_type += 1
                    binary_list_positive.append(1)
                else:
                    binary_list_positive.append(0)
                list_connector_value.append(dico_node_rs[node])
            print(list_connector_value)
            return [nb_connector_type / nb_connector_total,
                    len(positive_cell) / nb_total_nuclei,
                    list_connector_value,
                    binary_list_positive]
    elif len(positive_cell) == 0 and len(list_connector_index_value) > 0:
        return [0,
                len(positive_cell) / nb_total_nuclei,
                ["not implemented"],
                 []
        ]
    else:
        return None
    return None


def processing_fct_community_celltype_nb(dico_comm,
                                         dye_type,
                                         key_cell_name,
                                         dico_type = None):
    """
    Parameters
    ----------
    dico_comm
    dye_type
    key_cell_name
    dico_type
    Returns
    -------
    """


    dico_current_file = dico_comm[key_cell_name] # [nb af568, nb af647, total_nb_cell, clustering coeff]

    if dico_current_file == ["type no available ?"]:
        print("here %s"  % key_cell_name)
        return None

    if dye_type == "Cy3":
            # af568
            nb_type_index = 0
    else:
            # af647
            nb_type_index = 1
    value = []
    for commu in dico_current_file:
        percentage = dico_current_file[commu][nb_type_index] / dico_current_file[commu][2]
        if percentage > 0:
            value.append([percentage, dico_current_file[commu][nb_type_index],
                          dico_current_file[commu][2], dico_current_file[commu][3]])

    if len(value)==0:
        return None

    return  value

#%%

if __name__ == "__main__":

#%%
    folders_list = ["210426_repeat3/",  # ok
                    "200828-NIvsIR5M/00_Capillary_EC/",  # okd
                    "200828-NIvsIR5M/00_Large_Vessels/",  # okd
                    "200828-NIvsIR5M/00_Macrophages/",  # okd
                    "200908_CEC/",  # okd
                    "201030_fridyay/",  # ok
                    "210219_myo_fibros_y_macrophages/",  # okd
                    "210412_repeat_fibro/IR5M/",  # okd
                    "210412_repeat_fibro/NI/",  # okd
                    "210413_rep2/",  # okd
                    "210425_angiogenesis/",  # ok
                    "200908_fibrosis/",  # ok
                    "201127_AM_fibro/",  # okd
                   "210428_IR5M1236_Lamp3-Cy5_Pdgfra-Cy3/",
                   # "210205_Prolicence/aCap prolif/",
                   # "210205_Prolicence/aCap senes/",
                   # "210205_Prolicence/gCap prolif/",
                   # "210205_Prolicence/gCap senes/",
                "210205_Prolicence_old_for/aCap_prolif/",
                "210205_Prolicence_old_for/aCap_senes/",
                    ]


    for folder in folders_list:

        script_compute_general(folder = "/media/tom/Elements/to_take/" + folder,
                script_processing_fct = script_processing_fct_empty,
                name_ad = "empty2608",
                name_dico_type = "dico_stat_2106.npy",
                name_save = "dico_empty_neighbour2")


    list_probes  = [
        ['Lamp3'], ['Pdgfra'],
        ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
        ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
        ['Serpine1'], ['Apln'],
        ['Pecam1']
    ]

    for probe in list_probes:
        compute_gene(gene = probe,
            folders_list = folders_list,
            compute_gene_fct = compute_gene_fct_compute_empty_neighbour,
            path_to_take="/media/tom/Elements/to_take/",
            dico_name="empty_neighbour2",
            folder_save="empty_neighbour")

    for folder in folders_list:
        script_compute_general("/media/tom/Elements/to_take/" + folder,
                        script_processing_fct = script_processing_fct_degree,
                        name_ad = "dd_t5_2907",
                        name_dico_type = "dico_stat_2106.npy",
                        name_save = "dico_node_dee")

    list_probes  = [
        ['Lamp3'], ['Pdgfra'],
        ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
        ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
        ['Serpine1'], ['Apln'],
        ['Pecam1']
    ]
    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            gene,
            folders_list,
            compute_gene_fct=compute_gene_fct_degree,
            path_to_take="/media/tom/Elements/to_take/" ,
            dico_name="node_dee",
            folder_save="node_dee",

            dico_stat_name="dico_stat_2106")
#%%

    def mm(folder):
        script_compute_general("/media/tom/Elements/to_take/" + folder,
                               script_processing_fct=script_processing_fct_expected_edge,
                               name_ad="dd_t5_2907",
                               name_dico_type="dico_stat_2106.npy",
                               name_save="dico_expected_edges_maslov_sneppen")

    import multiprocessing

    number_processes = 6
    pool = multiprocessing.Pool(number_processes)
    results = pool.map_async(mm, folders_list)
    pool.close()
    pool.join()
#configuration

    for folder in folders_list:
        script_compute_general("/media/tom/Elements/to_take/" + folder,
                               script_processing_fct=script_processing_fct_expected_edge,
                               name_ad="dd_t5_2907",
                               name_dico_type="dico_stat_2106.npy",
                               name_save="dico_expected_edges_maslov_sneppen")

    list_probes  = [
                ['Lamp3'], ['Pdgfra'],
                ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
                ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
                ['Serpine1'], ['Apln'],
                ['Pecam1']
            ]


    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            gene,
            folders_list =folders_list,
            compute_gene_fct = compute_gene_fct_compute_expected_edge,
            path_to_take = "/media/tom/Elements/to_take/",
            dico_name = "expected_edges_maslov_sneppen",
            folder_save ="expected_edges_maslov_sneppen" ,
            dico_stat_name = "dico_stat_2106")



    list_probes  = [
                ['Lamp3'], ['Pdgfra'],
                ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
                ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
                ['Serpine1'], ['Apln'],
                ['Pecam1']
            ]


    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            [['Serpine1'] , gene],
            folders_list =folders_list,
            compute_gene_fct = compute_gene_fct_compute_expected_edge,
            path_to_take = "/media/tom/Elements/to_take/",
            dico_name = "expected_edges_configuration",
            folder_save ="expected_edge_configuration_model" ,
            dico_stat_name = "dico_stat_2106")


    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            [['Mki67'] , gene],
            folders_list =folders_list,
            compute_gene_fct = compute_gene_fct_compute_expected_edge,
            path_to_take = "/media/tom/Elements/to_take/",
            dico_name = "expected_edges_configuration",
            folder_save ="expected_edge_configuration_model" ,
            dico_stat_name = "dico_stat_2106")

#%%
   list_probes  = [
                ['Lamp3'], ['Pdgfra'],
                ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
                ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
                ['Serpine1'], ['Apln'],
                ['Pecam1']
            ]
    list_probes  = [[ ['Serpine1'], l] for l in list_probes]
    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            gene,
            folders_list=folders_list,
            compute_gene_fct=compute_gene_fct_compute_expected_edge,
            path_to_take="/media/tom/Elements/to_take/",
            dico_name="expected_edges",
            folder_save="expected_edge_configuration_model",
            dico_stat_name="dico_stat_2106")

   list_probes  = [
                ['Lamp3'], ['Pdgfra'],
                ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
                ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
                ['Serpine1'], ['Apln'],
                ['Pecam1']
            ]
    list_probes  = [[ ['Mki67'], l] for l in list_probes]
    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            gene,
            folders_list=folders_list,
            compute_gene_fct=compute_gene_fct_compute_expected_edge,
            path_to_take="/media/tom/Elements/to_take/",
            dico_name="expected_edges",
            folder_save="expected_edge_configuration_model",
            dico_stat_name="dico_stat_2106")

 #%% compute community features for each folder

    # folders_list = ["210425_angiogenesis/"]
    for name in folders_list:
        parser = argparse.ArgumentParser(description='generate_graph')
        parser.add_argument('-ptt', "--path_to_take", type=str,
                            default="/media/tom/Elements/to_take/",
                            help='path_to_take')

        parser.add_argument('-ptf', "--path_to_folder", type=str,
                            default=name,
                            help='path_to_take')
        parser.add_argument("--port", default=39949)
        parser.add_argument("--mode", default='client')

        args = parser.parse_args()
        # print(args)

        script_compute_general(
            folder=args.path_to_take + args.path_to_folder,
            script_processing_fct=script_processing_fct_community_features,
            name_ad="dd_t5_2907",
            name_dico_type="dico_stat_2106.npy",
            name_save="dico_comm_features_0410")
#%% compute community features for each gene
    list_probes  = [
            ['Lamp3'], ['Pdgfra'],
            ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
            ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
            ['Serpine1'], ['Apln'],
            ['Pecam1']
        ]
    for probe in list_probes:

        compute_gene(gene=probe,
                        folders_list=folders_list,
                        compute_gene_fct=processing_fct_community_celltype_nb,
                        path_to_take="/media/tom/Elements/to_take/",
                        dico_name="comm_features_0410",
                        folder_save= "community_celltype_nb",
                        dico_stat_name="dico_stat_2106")

#%%
    for folder in folders_list:
        script_compute_general("/media/tom/Elements/to_take/" + folder,
                        script_processing_fct = script_processing_fct_degree,
                        name_ad = "dd_t5_2907",
                        name_dico_type = "dico_stat_2106.npy",
                        name_save = "dico_node_degree")

#%%
    list_probes  = [
        ['Lamp3'], ['Pdgfra'],
        ['Chil3'], ['Cap', 'aCap', 'CEC'], ['Ptprb'],
        ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'],
        ['Serpine1'], ['Apln'],
        ['Pecam1']
    ]
    for gene in list_probes:
        compute_gene(  ### FORMER compute_general
            gene,
            folders_list,
            compute_gene_fct=compute_gene_fct_degree,
            path_to_take="/media/tom/Elements/to_take/" ,
            dico_name="node_degree",
            folder_save="node_degree_",
            dico_stat_name="dico_stat_2106")


    #%%
script_compute_general(folder,
                script_processing_fct = script_compute_empty_contact_without_type,
                name_ad = "dd_t5_2907",
                name_dico_type = "dico_stat_2106.npy",
                name_save = "dico_empty_neighbour")






#%% generate dico_edge



    for name in folders_list:
        parser = argparse.ArgumentParser(description='generate_graph')
        parser.add_argument('-ptt', "--path_to_take", type=str,
                            default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
                            help='path_to_take')


        parser.add_argument('-ptf', "--path_to_folder", type=str,
                            default=name,
                            help='path_to_take')
        parser.add_argument("--port", default=39949)
        parser.add_argument("--mode", default='client')

        args = parser.parse_args()
        #print(args)

        script_compute_nb_edge(folder = args.path_to_take + args.path_to_folder, name_ad="dd_t5_2907")

#%% generate dico_ratio for each gene



    for probe in list_probes:
        compute_expected_edge(gene = probe,
                 folders_list = folders_list,
                 path_to_take = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/")



#%% generate dico_volume_degree for each folder

    for name in folders_list:
        parser = argparse.ArgumentParser(description='generate_graph')
        parser.add_argument('-ptt', "--path_to_take", type=str,
                            default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
                            help='path_to_take')


        parser.add_argument('-ptf', "--path_to_folder", type=str,
                            default=name,
                            help='path_to_take')
        parser.add_argument("--port", default=39949)
        parser.add_argument("--mode", default='client')

        args = parser.parse_args()
        #print(args)

        script_compute_degree_volume(folder = args.path_to_take + args.path_to_folder, name_ad="dd_t5_2907")

#%% generate dico_volume_degree for each gene

    list_probes  = [
        ['Lamp3'], ['Pdgfra'],
        ['Chil3'], ['Cap', 'aCap'], ['Ptprb'],
               ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'], ['Serpine1'], ['Apln'],
               ['Pecam1'], ['CEC']]

    for probe in list_probes:

        compute_general(gene = probe,
                 folders_list = folders_list,
                        processing_fct = processing_fct_compute_degree_volume,
                        path_to_take="/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
                        dico_name="dico_degree_volume",
                        folder_save="degree_volume")


#%% compute rao_stirling index for each folder


   # folders_list = ["210425_angiogenesis/"]
    for name in folders_list:
        parser = argparse.ArgumentParser(description='generate_graph')
        parser.add_argument('-ptt', "--path_to_take", type=str,
                            default="/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
                            help='path_to_take')


        parser.add_argument('-ptf', "--path_to_folder", type=str,
                            default=name,
                            help='path_to_take')
        parser.add_argument("--port", default=39949)
        parser.add_argument("--mode", default='client')

        args = parser.parse_args()
        #print(args)

        script_compute_general(
folder = args.path_to_take + args.path_to_folder,
script_processing_fct = script_processing_fct_rao_stirling,
name_ad="dd_t5_2907",
name_dico_type = "dico_stat_2106.npy",
name_save = "dico_rao_stirling")


#%% compute rao_stirling index for each gene

list_probes  = [
        ['Lamp3'], ['Pdgfra'],
        ['Chil3'], ['Cap', 'aCap','CEC'], ['Ptprb'],
               ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'], ['Serpine1'], ['Apln'],
               ['Pecam1'], ]

for probe in list_probes:
    compute_gene(gene=probe,
                    folders_list=folders_list,
                    compute_gene_fct=processing_fct_proportion_rao_stirling_index,
                    path_to_take="/media/tom/Elements/to_take/",
                    dico_name="rao_stirling",
                    folder_save="rao_stirling_prp",
                    dico_stat_name="dico_stat_2106")


#%% garbage


########
# script function for script_compute_general processing_fct
#######



def script_compute_empty_contact(folder, name_ad = "2405_wt0", name_dico_type = "dico_stat_2106.npy"):
    """
    Compute the percentage of cell positive to gene that are in contact with empty space

    Parameters
    ----------


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
    dico_empty = {}
    for f in onlyfiles:
        try:
            dico_nb_with_zero = remove_self_loop_dico(np.load(path_to_ad +  "dico_no" + name_ad + f + ".npy", allow_pickle=True).item())
            positive_af568 = [dico_type[f][5][i][3] for i in range(len(dico_type[f][5]))]
            positive_af647 = [dico_type[f][6][i][3] for i in range(len(dico_type[f][6]))]
            per_af568 , per_af647 , nb_af568, nb_af647 = compute_nb_empty_contact(dico_nb_with_zero, positive_af568, positive_af647)
            dico_empty[f] = [("per_af568", per_af568), ("per_af647", per_af647), ("total_af568", nb_af568), ("total_af647", nb_af647)]
        except Exception as e:
            print(e)
            dico_empty[f] = ["type no available ?"]
    np.save(folder + "dico_empty", dico_empty)
    return dico_empty

def script_compute_empty_contact_without_type(folder, name_ad = "2405_wt0", name_dico_type = None):
    """
    Compute the percentage of cell positive to gene that are in contact with empty space

    Parameters
    ----------
    gene
    folders_list
    path_to_take

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
    dico_empty = {}
    for f in onlyfiles:
        try:
            dico_nb_with_zero = remove_self_loop_dico(np.load(path_to_ad +  "dico_no" + name_ad + f + ".npy", allow_pickle=True).item())
            proportion, nb_zero = compute_nb_empty_contact(dico_nb_with_zero,
                                                            positive_af568=None,
                                                            positive_af647 = None)
            dico_empty[f] = [("proportion", proportion), ("nb_zero", nb_zero)]
        except Exception as e:
            print(e)
            dico_empty[f] = ["type/data no available ?"]
    np.save(folder + "dico_empty", dico_empty)
    return dico_empty


def script_compute_degree_volume(folder, name_ad = "dd_t5_2907", name_dico_type = "dico_stat_2106.npy"):
    path_output_segmentaton = folder + "tiff_data/predicted_mask_dapi/"
    onlyfiles = [f for f in listdir(path_output_segmentaton) if
                 isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

    path_to_ad = folder + "adjacent_list/"
    dico_type = np.load(folder + name_dico_type,
                        allow_pickle=True).item()
    dico_degree_volume = {}
    for f in onlyfiles:
        try:
        #                    dico_stat[f] = [len(np.unique(img_dapi_mask))0, nb_no_rna1, nb_cy32, nb_cy53, nb_both4,
        #                            positive_cluster_5685, positive_cluster_6476, negative_cluster_568, negative_cluster_647]
        # positive_cluster.append([cluster0, overlap1, ConvexHull(cluster_spots).volume2, cs3, ConvexHull(cluster_spots).area, longuest_distance])
            positive_cluster_568 = dico_type[f][5]
            positive_cluster_647 = dico_type[f][6]
            dico_nb = remove_self_loop_dico(np.load(path_to_ad + "dico_no" + name_ad + f + ".npy", allow_pickle=True).item())
            dico_degree_volume[f] = [[[pos568[3], pos568[2],
                                     len(dico_nb[pos568[3]]), "af568"] for pos568 in positive_cluster_568] ,
                                     [[pos647[3], pos647[2],
                                     len(dico_nb[pos647[3]]), "af64"] for pos647 in positive_cluster_647]]
        except Exception as e:
            print(e)
            dico_degree_volume[f] = ["type no available ?"]
    np.save(folder + "dico_degree_volume", dico_degree_volume)
    return dico_degree_volume