
#from matplotlib import pyplot as plt

import os
from os import listdir
from os.path import isfile, join
#from matplotlib import pyplot as plt
import tifffile
import numpy as np
from cellpose import models, io, plot

#from skimage.exposure import rescale_intensity

from pathlib import Path
import argparse
#from cellpose import metrics


def erase_solitary(mask): #mask en 3D
    mask_bis = np.zeros(mask.shape)

    ##traiter le cas ou n = 1
    current_nuclei = set(np.unique(mask[0]))
    post_nuclei = set(np.unique(mask[1]))
    nuclei_to_remove =  current_nuclei - post_nuclei
    nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
    for nuc in nuclei_to_keep:
        mask_bis[0] += (mask[0] == nuc) * mask[0]

    for i in range(1, len(mask)-1):
        pre_nuclei = set(np.unique(mask[i-1]))
        current_nuclei = set(np.unique(mask[i]))
        post_nuclei = set(np.unique(mask[i+1]))
        nuclei_to_remove =  current_nuclei - pre_nuclei - post_nuclei
        nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
        for nuc in nuclei_to_keep:
            mask_bis[i] += (mask[i] == nuc) *  mask[i]
    ##traiter le cas ou n = -1
    current_nuclei = set(np.unique(mask[-1]))
    pre_nuclei = set(np.unique(mask[-2]))
    nuclei_to_remove =  current_nuclei - pre_nuclei
    nuclei_to_keep = current_nuclei - nuclei_to_remove # reminder: set operation are different from arithemtic operation
    for nuc in nuclei_to_keep:
        mask_bis[-1] += (mask[-1] == nuc) * mask[-1]
    return mask_bis

def stitch3D(masks, stitch_threshold=0.25):
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU
    from utils_ext.cellpose_utilis import stitch3D
    """
    from cellpose import metrics

    mmax = masks[0].max()
    for i in range(len(masks)-1):
        try:
            iou = metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1)==0.0)[0]
            istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i+1] = istitch[masks[i+1]]
        except:
            continue
    return masks

def divide_dapi_fish_tiff(path_raw_tiff = "/home/tom/Bureau/stiched_images_florian/old_folder/path_raw"):
    p = Path(path_raw_tiff)
    list_path_tiff = list(p.glob('*'))
    print(list_path_tiff)

    if not os.path.exists(path_raw_tiff + "fish/"):
        os.mkdir(path_raw_tiff + "fish/")
    if not os.path.exists(path_raw_tiff + "dapi/"):
        os.mkdir(path_raw_tiff + "dapi/")
    for lp in  list_path_tiff:
        list_tiff = list(lp.glob('*ome.tif'))
        print(list_tiff)
        for lt in list_tiff:
            double_image = tifffile.imread(str(lt))
            print(str(lt))
            print(double_image.shape)
            if double_image.shape != (2, 53, 2048, 2048) and double_image.shape != (53, 2, 2048, 2048):
                if double_image.shape == (106, 2048, 2048):
                    fish  = double_image[:53]
                    dapi =  double_image[53:]
            elif  double_image.shape == (53, 2, 2048, 2048):
                double_image  = double_image.transpose(1,0,2,3)
                fish = double_image[0]
                dapi = double_image[1]
            else:
                assert  double_image.shape == (2, 53, 2048, 2048)
                fish = double_image[0]
                dapi = double_image[1]
            name_image = str(lt).split('/')[-1][:-8]
            tifffile.imsave(path_raw_tiff + "fish/" + name_image +".tiff", fish)
            tifffile.imsave(path_raw_tiff + "dapi/" + name_image+".tiff", dapi)


def segment_nuclei(path_to_dapi, path_to_mask_dapi, dico_param, model, save=True):
    if not os.path.exists(path_to_mask_dapi):
        os.mkdir(path_to_mask_dapi)
    onlyfiles = [f for f in listdir(path_to_dapi) if isfile(join(path_to_dapi, f))]
    print(onlyfiles)
    dico_res = {}
    for f in onlyfiles:
        print(f)
        img = tifffile.imread(path_to_dapi + f)
        print(img.shape)
        if dico_param["mip"] is True and len(img.shape) == 3:
            img = np.amax(img, 0)
        # elif dico_param["mip"]
        else:
            assert len(img.shape) == 3
            img_shape = img.shape
            img = img.reshape(img_shape[0], 1,img_shape[1], img_shape[2])
            img = list(img)
        masks, flows, styles, diams = model.eval(img, diameter=dico_param["diameter"],
                                                 channels=[0, 0],
                                                 flow_threshold=dico_param["flow_threshold"],
                                                # mask_threshold=dico_param["mask_threshold"],
                                                 do_3D=dico_param["do_3D"],
                                                 stitch_threshold=0)
        masks = stitch3D(masks, dico_param["stitch_threshold"])
        masks = np.array(masks, dtype = np.int16)
        if args.erase_solitary:
            masks = erase_solitary(masks)
        dico_res[f] = [masks, flows, styles, diams]
        if save:
            if path_to_mask_dapi[-1] == '/':
                tifffile.imwrite(path_to_mask_dapi + "dapi_mask" + f, data=masks, dtype=masks.dtype)
                np.save(path_to_mask_dapi + "dico_param.npy", dico_param)
            else:
                tifffile.imwrite(path_to_mask_dapi + "/dapi_mask" + f, data=masks, dtype=masks.dtype)
                np.save(path_to_mask_dapi + "/dico_param.npy", dico_param)
    #np.save(path_to_mask_dapi + "/dico_res.npy", dico_res)
    return dico_res

#%%
if __name__ == '__main__':
    # function that check the shape and split the tiff in nuclei vs RNA.

    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-ptd', "--path_to_dapi",
                        type=str,
                        default="/media/tom/Transcend41/to_transf/2022-06-09_benchmark-topo/acquisition/multi-positions/r_beads_50ms_1/dapi/r/",
                        help='path_to_czi folder')

    parser.add_argument('-ptm', "--path_to_mask_dapi",
                        type=str,
                        default="/media/tom/Transcend41/to_transf/2022-06-09_benchmark-topo/acquisition/multi-positions/r_beads_50ms_1/dapi_mask/",
                        help=' folder')

    parser.add_argument('-g', "--gpu", type=int, default=1, help='')
    parser.add_argument('-d', "--diameter", type=float, default=None, help='')
    parser.add_argument('-ft', "--flow_threshold", type=float, default=0.75, help='')
    parser.add_argument("--mask_threshold", type=float, default=0, help='')
    parser.add_argument('-d3', "--do_3D", type=bool, default=False, help='')
    parser.add_argument('-m', "--mip", type=bool, default=False, help='')
    parser.add_argument('-st', "--stitch_threshold", type=float, default=0.3, help='')
    parser.add_argument('-er', "--erase_solitary", type=int, default=0, help='')
    parser.add_argument('-mt', "--model_type", type=str, default="nuclei", help='')
    parser.add_argument("--port", default=39949)
    parser.add_argument("--mode", default='client')
    args = parser.parse_args()
    print(args)

    model = models.Cellpose(gpu=args.gpu, model_type=args.model_type)
    dico_param = {}
    dico_param["diameter"] = args.diameter
    dico_param["flow_threshold"] = args.flow_threshold
    dico_param["mask_threshold"] = args.mask_threshold
    dico_param["do_3D"] = args.do_3D
    dico_param["mip"] = args.mip
    dico_param["projected_focused"] = False
    dico_param["stitch_threshold"] = args.stitch_threshold

    for i in range(80, 90, 10):
        print(i)
        dico_param["diameter"] = i
        dico_res = segment_nuclei(args.path_to_dapi, args.path_to_mask_dapi + "_"+str(dico_param["diameter"] )+"ft"+str(args.flow_threshold), dico_param, model, save=True)



#%%
from spots.spot_detection import cluster_over_nuclei_3D_convex_hull, spot_detection_for_clustering
