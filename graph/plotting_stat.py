
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.stats import mannwhitneyu

def get_mouse_name(image_name):
    if any(word in image_name for word in ["NI", 'Ctrl']):
        if all(word in image_name for word in ["1225"]):
            return "1225"
        elif all(word in image_name for word in ["1230"]):
            return "1230"
        else:
            return "2323"

    if any(word in image_name for word in ['IR5M']):
        if "1236" in image_name:
            return "1236"
        elif "1249" in image_name:
            return "1249"
        elif "1250" in image_name:
            return "1250"
        elif "2330" in image_name:
            return "2330"
        else:
            return "2201"



def sort_list_by_name(list_tuple):
    list_NI = []
    list_IRM = []
    dico_name = {}
    for name in ["1230", "1225", "2323", "1236", "1249", "1250", "2330", "2201"]:
        dico_name[name] = []
    for tup in list_tuple:
        dico_name[tup[1]].append(tup[0])
    list_NI = [[dico_name["1230"], "1230"], [dico_name["1225"], "1225"], [dico_name["2323"], "2323"]]

    list_IR5M = [[dico_name["1236"], "1236"], [dico_name["1249"], "1249"],
                 [dico_name["1250"], "1250"], [dico_name["2330"], "2330"],
                 [dico_name["2201"], "2201"]]
    return list_NI, list_IR5M


def plot_scatter(list_NI, list_IRM, dico_color, cell_name, save_path,
                 title="average size of point cloud per sample for ",
                 axis_y_label="size of point cloud in  μm3"):
    """
    list_NI is a list of list [list_value, name]
    dico_color = {
    "1225": '#1f77b4',
    "1230": '#ff7f0e',
    "2323": '#2ca02c',

    "1236": '#d62728',
    "1249": '#9467bd',
    "1250": '#8c564b',
    "2201": '#e377c2',
    "2330": '#7f7f7f',
    }

    dico_label = {
    "1225": "NI_1225",
    "1230": 'NI_1230',
    "2323": 'NI_2323',

    "1236": 'IR5M_1236',
    "1249": 'IR5M_1249',
    "1250": 'IR5M_1250',
    "2201": 'IR5M_2201',
    "2330": 'IR5M_2330',
    }
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    np.random.seed(seed=40)
    x_tick = []
    x_labels = []
    x_index = 1
    for l in list_NI:
        if len(l[0]) == 0:
            continue
        y = l[0]
        x = [x_index] * len(l[0]) + np.random.rand(len(l[0])) - 0.5
        s, c = np.array([2.0] * len(l[0])), np.array([dico_color[l[1]]] * len(l[0]))
        s *= 10.

        ax.scatter(x, y, s, c, label=dico_label[l[1]])
        x_tick.append(x_index)
        x_labels.append("NI " + l[1])
        # x_index += 1

    if len(x_tick) > 0:
        median_NI = np.median([k for el in [l[0] for l in list_NI] for k in el])
        mean_NI = np.mean([k for el in [l[0] for l in list_NI] for k in el])
        xmax_NI = x_tick[-1]
        ax.hlines(y=median_NI, xmin=0.5, xmax=xmax_NI + 0.5, linewidth=3)
        print(median_NI)

    else:
        xmax_NI = 0
        mean_NI = 0
        median_NI = 0
    x_index += 1.3

    for l in list_IRM:
        if len(l[0]) == 0:
            continue
            print(x_index)
        y = l[0]
        x = np.array([x_index] * len(l[0])) + np.random.rand(len(l[0])) - 0.5
        s, c = np.array([2.0] * len(l[0])), np.array([dico_color[l[1]]] * len(l[0]))
        s *= 10.

        ax.scatter(x, y, s, c, label=dico_label[l[1]])
        x_tick.append(x_index)
        x_labels.append("IR5M " + l[1])
        # x_index += 1

    median_IRM = np.median([k for el in [l[0] for l in list_IRM] for k in el])
    mean_IRM = np.mean([k for el in [l[0] for l in list_IRM] for k in el])

    ax.hlines(y=median_IRM, xmin=xmax_NI + 1.8, xmax=x_index - 0.5, linewidth=3)

    # draw vertical line from (70,100) to (70, 250)
    ax.set_xticks([1, 2])

    ax.set_xticklabels(["NI", "IR5M"], rotation=45)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=15)
    #  ax.set_xlabel("Animals", fontsize = 20)
    ax.set_ylabel(axis_y_label, fontsize=15)
    fig.suptitle(title, fontsize=20)
    try:
        U1, p = mannwhitneyu([k for el in [l[0] for l in list_IRM] for k in el],
                             [k for el in [l[0] for l in list_NI] for k in el])

        ax.set_title('P value of the Mann-Whitney test: ' + str(round(p, 6)).ljust(6, '0'), fontsize=15)
    except Exception as e:
        ax.set_title(str(e), fontsize=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    fig.savefig(save_path + cell_name, bbox_inches='tight')
    return round(mean_NI, 4), round(median_NI, 4), round(mean_IRM, 4), round(median_IRM, 4)


def plot_edge_ratio(dico_cellname_probe, list_probes, path_to_dico):
    """
    list_probes  = [ ['Lamp3'],  ['Pdgfra'],['Serpine1'], ['Ptprb'],['Apln'], ['Chil3'],  ['Fibin'],
                ['C3ar1'],  ['Hhip'], ['Mki67'],
           ['Pecam1'],  ['Cap', 'aCap', 'CEC'],
           ]


    dico_cellname_probe = {   'Lamp3':"AT2", 'Pdgfra':"fibroblast",
    "Serpine1": "scenescent",  'Ptprb': "vessel EC",
    'Apln': "capillary EC with Alpn",  'Chil3': "AM", "Fibin":"capillary EC with Fibin",
    'C3ar1':"IM", 'Hhip': "myifibroblast", 'Mki67':"cycling cells",
    'Pecam1': "EC",'Cap':"capillary EC",'aCap':"capillary EC",'CEC':"capillary EC",}

    path_to_dico = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/stat_edge/"
    """

    for probe in list_probes:
        dico_NI = np.load(path_to_dico + probe[0] + "NI.npy", allow_pickle = True).item()
        dico_IR5M = np.load(path_to_dico + probe[0] + "IR5M.npy", allow_pickle = True).item()
        list_NI = [[dico_NI[k][-1], get_mouse_name(k)]  for k in dico_NI]
        list_IR5M = [[dico_IR5M[k][-1], get_mouse_name(k)] for k in dico_IR5M]
        cleanedlist_IR5M = [x for x in list_IR5M if x[0] == x[0]]
        cleanedlist_NI = [x for x in list_NI if x[0] == x[0]]
        list_NI, list_IR5M = sort_list_by_name(cleanedlist_IR5M + cleanedlist_NI)
        mean_NI, median_NI, mean_IRM, median_IRM = plot_scatter(list_NI, list_IR5M, dico_color, cell_name= dico_cellname_probe[probe[0]],
        save_path = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/stat_edge/"+probe[0],
         title = 'edge ratio for ' +  dico_cellname_probe[probe[0]] ,
        axis_y_label = "nb_edge/(nb_expected_edge)")


#%% scripting zone : compute louvain partition for all graph.

from graph.toolbox import get_louvain_partition

path_to_dico = "/home/tom/Bureau/annotation/cell_type_annotation/to_take/stat_edge/"
dico_NI = np.load(path_to_dico + probe[0] + "NI.npy", allow_pickle=True).item()
dico_IR5M = np.load(path_to_dico + probe[0] + "IR5M.npy", allow_pickle=True).item()
list_NI = [[dico_NI[k][-1], get_mouse_name(k)] for k in dico_NI]
list_IR5M = [[dico_IR5M[k][-1], get_mouse_name(k)] for k in dico_IR5M]
cleanedlist_IR5M = [x for x in list_IR5M if x[0] == x[0]]
cleanedlist_NI = [x for x in list_NI if x[0] == x[0]]
list_NI, list_IR5M = sort_list_by_name(cleanedlist_IR5M + cleanedlist_NI)