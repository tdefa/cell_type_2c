# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import tifffile
import scipy.stats
from post_processing import erase_solitary, erase_small_nuclei

import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt




def plot_marked_volume(marked_image_volume, mask):
   for slice in range(len(marked_image_volume)):
       if np.count_nonzero(mask[slice,:,:]):
           plt.figure(figsize=(10,10))
           edges_pz = mark_boundaries(marked_image_volume[slice,:,:], mask[slice].astype(np.int),
                                                color=(1,0,0), mode='thin')
           plt.imshow(edges_pz)
           plt.title('slice ' + str(slice))
           plt.show()


def evaluate_iou(path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14",
                 path_prediction = "/home/thomas/Bureau/phd/first_one/tiff_data/predicted_mask_dapi_st04", d3 = True, iou_thresh = 0.5, erase_so = True):


    path_label = Path(path_label)
    path_prediction = Path(path_prediction)
    ap_list = []
    ac_list =[]
    for image_files_label in path_label.glob("*/"):
       # print(image_files_label)
        image_label = imread(image_files_label)
        try:
            slice_label = int(list(image_files_label.parts)[-1][-6:-4])
            image_name = str(list(image_files_label.parts)[-1][:-9])
        except :
            slice_label = int(list(image_files_label.parts)[-1][-5:-4])
            image_name = str(list(image_files_label.parts)[-1][:-8])
        if erase_so:
            image_pred =  erase_solitary(tifffile.imread(str(path_prediction) +"/dapi_maskdapi_" +  image_name  + ".tiff"))
        else:
            image_pred =  tifffile.imread(str(path_prediction) +"/dapi_maskdapi_" +  image_name  + ".tiff")
        cell_label = np.unique(image_label)
        cell_pred = np.unique(image_pred[slice_label])
        dico_ap = {}
        for c in cell_label:
            current_cell = (image_label == c).astype(bool)
            for pred in cell_pred:
                overlap = current_cell * (image_pred[slice_label] == pred).astype(bool) # Logical AND
                union = current_cell +  (image_pred[slice_label] == pred).astype(bool)# Logical OR
                if overlap.sum()/float(union.sum()) > iou_thresh:
                    dico_ap[c] = pred
        tp = len(dico_ap)
        fp = len(cell_pred)  - len(dico_ap.values()) #len([cell_pred[i] for i in range(len(cell_pred)) if cell_pred[i] not in dico_ap.values()])
        fn = len(cell_label) - len(dico_ap.keys()) #len([cell_label[i] for i in range(len(cell_label)) if cell_label[i] not in dico_ap.keys()])
    
        ap = tp / (tp+fp+fn)
        ap_list.append(ap)
    return ap_list


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def generate_datset():
    path_save = Path(r'/home/thomas/Bureau/phd/kaibu_data/dandra_3d_14')   # Folder to store data
    path_new_dataset = Path(r'/home/thomas/Bureau/phd/label_dataset/sandra_3d_v0')
    for folder in path_save.glob("*/"):
        print(folder)
        image = folder / "target_files_v0/annotation.png"
        if not image.is_file():
            print("t")
            image = folder / "target_files_v0/annotation.png"
        image = imread(image)

        imsave(str(path_new_dataset / list(folder.parts)[-1]) + ".png", image)

    plt.imshow(image)



def from_labe_to_training(): #from label dataset to dataset for training;
    path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14"
    path_input = "/home/thomas/Bureau/phd/first_one/tiff_data/dapi"
    path_label = Path(path_label)
    path_input = Path(path_input)


#%%

if __name__ == "__main__":
    path_label = "/home/thomas/Bureau/phd/label_dataset/dandra_3d_14"
    path_input = "/home/thomas/Bureau/phd/first_one/tiff_data/dapi"
    path_to_save = "/home/thomas/Bureau/phd/label_dataset/with_name/"
    path_label = Path(path_label)
    i = 0
    for image_files_label in path_label.glob("*/"):
        print(i)
        image_label = imread(image_files_label)
        try:
            slice_label = int(list(image_files_label.parts)[-1][-6:-4])
            image_name = str(list(image_files_label.parts)[-1][:-9])
        except :
            slice_label = int(list(image_files_label.parts)[-1][-5:-4])
            image_name = str(list(image_files_label.parts)[-1][:-8])
        image_input =  tifffile.imread(str(path_input +"/dapi_" +  image_name  + ".tiff"))
        np.save(path_to_save + image_name +"_"+ str(slice_label).zfill(2) + '_img',image_label )
        np.save(path_to_save + image_name +"_"+ str(slice_label).zfill(2) + '_mask',image_input[slice_label])
        i += 1

#%%
