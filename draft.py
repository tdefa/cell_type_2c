

import argparse
import os
from os import listdir
from os.path import isfile, join
import czifile as zis
from matplotlib import pyplot as plt
import tifffile
import numpy as np


path_to_czi = "/media/tom/Elements/to_take/200710_Round2/"
path_to_dapi = "/media/tom/Elements/to_take/200710_Round2/tiff_data/dapi/"
path_to_af647 = "/media/tom/Elements/to_take/200710_Round2/tiff_data/af568/"
path_to_af568 = "/media/tom/Elements/to_take/200710_Round2/tiff_data/af647/"
if not os.path.exists(path_to_czi + "tiff_data/"):
    os.mkdir(path_to_czi + "tiff_data/")
if not os.path.exists(path_to_czi + "tiff_data/" + "dapi/"):
    os.mkdir(path_to_czi + "tiff_data/" + "dapi/")
if not os.path.exists(path_to_czi + "tiff_data/" + "af568/"):
    os.mkdir(path_to_czi + "tiff_data/" + "af568/")
if not os.path.exists(path_to_czi + "tiff_data/" + "af647/"):
    os.mkdir(path_to_czi + "tiff_data/" + "af647/")
onlyfiles = [f for f in listdir(path_to_czi) if isfile(join(path_to_czi, f)) and join(path_to_czi, f)[-3:] == "czi"]
for f in onlyfiles:
    print(f)
    czi = zis.CziFile(path_to_czi + f)
    metadatadict_czi = czi.metadata(raw=False)     # parse the XML into a dictionary
    chanel = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']
    chanel_name = [chanel[i]["Name"] for i in range(len(chanel))]
    print(chanel_name)
    array_im = zis.imread(path_to_czi + f)
    for i in range(len(chanel_name)):
        if chanel_name[i] == "Alexa Fluor 647" or chanel_name[i] == 'Cy5':
            array_af647_3d = array_im[0,0, i, :, :, :, 0]
            tifffile.imwrite(path_to_af647 + "AF647_" + f[:-3] + "tiff", data=array_af647_3d,
                             shape=array_af647_3d .shape, dtype=array_af647_3d.dtype)
        if chanel_name[i] == "Alexa Fluor 568" or chanel_name[i] == 'Cy3':
            array_af568_3d = array_im[0,0, i, :, :, :, 0]
            tifffile.imwrite(path_to_af568 + "AF568_" + f[:-3] + "tiff", data=array_af568_3d,
                             shape=array_af568_3d.shape, dtype=array_af568_3d.dtype)
        if chanel_name[i] == "DAPI":
            array_dapi_3d = array_im[0,0, i, :, :, :, 0]
            tifffile.imwrite(path_to_dapi + "dapi_" + f[:-3] + "tiff", data=array_dapi_3d,
                             shape=array_dapi_3d.shape, dtype=array_dapi_3d.dtype)



#%%
masks = tifffile.imread("/media/tom/Elements/to_take/200908_CEC/tiff_data/erase_solitary/dapi_maskdapi_05_IR4M_CEC-Cy3_Serpine1-Cy5_07.tiff")
dapi = tifffile.imread("/media/tom/Elements/to_take/200908_CEC/tiff_data/dapi/dapi_05_IR4M_CEC-Cy3_Serpine1-Cy5_07.tiff")
adjacent_list = np.load("/media/tom/Elements/to_take/200908_CEC/adjacent_list/ad_listdd_t5_290705_IR4M_CEC-Cy3_Serpine1-Cy5_07.tiff.npy")
plot_top_h_centrality_plus_edge(dapi, masks, dico_centrality = {},
                                    adjacent_list=adjacent_list , centroid=None, top=0)


#%%

import time
import tifffile
import bigfish.detection as detection
import bigfish.stack as stack
def spot_detection_for_clustering(sigma, rna_path, path_output_segmentaton,
                                  threshold_input = None,
                                  output_file = "detected_spot_3d/",
                                min_distance = (3,3,3),
                                  path_to_mask_dapi  = None):
    """
    function to detect the spots with a given sigma and return also the theshold
    Parameters
    ----------
    sigma
    float_out
    rna_path
    path_output_segmentaton
    threshold_input
    output_file
    path_to_mask_dapi

    Returns
    -------

    """
    dico_threshold = {}
    onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f)) and f[-1] == "f" ]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
    print(onlyfiles)
    for index_path in range(len(rna_path)):
        path = rna_path[index_path]
        for file_index in range(len(onlyfiles)):
            t = time.time()
            rna = tifffile.imread(path + onlyfiles[file_index])
            min_distance = min_distance
            print(sigma)
            rna_log = stack.log_filter(rna, sigma)#, float_out)
            # local maximum detection
            mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
            if threshold_input is not None and onlyfiles[file_index] in threshold_input:
                threshold = threshold_input[onlyfiles[file_index]]
                rna_log = stack.log_filter(rna, sigma, float_out = False)
                print("manuel threshold")
            else:
                threshold = detection.automated_threshold_setting(rna_log, mask)
            print(threshold)
            spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
            dico_threshold[onlyfiles[file_index]] = [threshold, len(spots)]
            np.save( output_file + path[-6:] + onlyfiles[file_index][:-5] + 'array.npy', spots)
            print(len(spots))
    return dico_threshold


import napari


import time
import tifffile
import numpy as np


fish1 = tifffile.imread("/media/tom/Elements1/to_take/201030_fridyay/tiff_data/af568/AF568_02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_005.tiff")
fish2 = tifffile.imread("/media/tom/Elements1/to_take/201030_fridyay/tiff_data/af647/AF647_02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_005.tiff")

dapi =  tifffile.imread("/media/tom/Elements1/to_take/201030_fridyay/tiff_data/predicted_mask_dapi/dapi_maskdapi_02_IR5M_Lamp3-Cy3_Pdgfra-Cy5_005.tiff")



viewer = napari.Viewer()
viewer.add_image(fish1, name = 'fish1', scale=(1, 0.3, 0.3))
viewer.add_image(fish2, name = 'fish2',scale=(1, 0.3, 0.3))

viewer.add_image(dapi, name = 'dapi', scale=(1, 0.3, 0.3))


viewer.add_points(spotst, size=3, scale=(3, 1, 1),
                  edge_color = "red",
                  face_color = "red",
                  ndim=3)


len(spotst)
0.000035
0.000035

0.000040

0.000050
0.000060
#%%
sigma = 1.35
rna = fish
min_distance = (3,3,3)
rna_log = stack.log_filter(rna, sigma)
# local maximum detection
mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
spots, _ = detection.spots_thresholding(rna_log, mask, threshold =0.000060)
print(len(spots))
if threshold_input is not None and onlyfiles[file_index] in threshold_input:
    threshold = threshold_input[onlyfiles[file_index]]
    rna_log = stack.log_filter(rna, sigma)
    print("manuel threshold")
else:
    threshold = detection.automated_threshold_setting(rna_log, mask)
spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
dico_threshold[onlyfiles[file_index]] = [threshold, len(spots)]


viewer = napari.Viewer()
viewer.add_image(fish, name = 'mask', scale=(3, 1, 1))
viewer.add_image(dapi, name = 'dapi', scale=(3, 1, 1))

viewer.add_points(spots, size=3, scale=(3, 1, 1),
                  edge_color = "red",
                  face_color = "red",
                  ndim=3)



#%%
np.save("/media/tom/Elements/to_take/211207_10gy/detected_spot_3d/AF647_IR5M10Gy_1259_Lamp3Cy5_Rtkn2Cy3_03array.npy", spots)
