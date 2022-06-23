
#%%
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import tifffile
import numpy as np
#from cellpose import models, io, plot
from skimage.exposure import rescale_intensity

from skimage.registration import phase_cross_correlation

from skimage.exposure import rescale_intensity

from pathlib import Path
import argparse


def naive_stich(dico_pos_im, OverlapPixels =93, final_shape = (53, 5865, 5865)):
    final_image = np.zeros(final_shape)
    for cord in dico_pos_im.keys():
        l,c = cord[0], cord[1]
        im = dico_pos_im[cord]
        print(cord)
        print(im[:, :-93, :-93 ].shape)
        final_image[:, l * 2048  - 93* l  : 2048 * (l+1) - 93* (l+1) ,
        c * 2048 - 93* c : 2048 * (c+1) - 93 * (c+1)] = im[:, :-93, :-93 ]

    return final_image


# OverlapPixels
def plot_rescal(final_image,  figsize=(50, 50)):
    input = np.amax(final_image, 0)
    pa_ch1, pb_ch1 = np.percentile(input, (1,99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(final_image_mip)
    plt.show()



def stitching_with_coordo(list_image, list_cordo):
    """
    Grid: row - by - row:
    right to down
    Parameters
    ----------
    list_image:
    nb_row
    nb_col
    list_cordo list of (x,y) coordinate of the top right corner of each image
    Returns
    -------
    """
    print("totdo add a blending ")
    list_cordo = np.array(list_cordo).round()
    list_cordo = list_cordo - np.array([min([x[0] for x in list_cordo]),
                                       min([x[1] for x in list_cordo])])
    image_lx = list_image[0].shape[-2]
    image_ly = list_image[0].shape[-1]

    final_shape_xy = np.array([np.max(list_cordo[:, 0]),np.max(list_cordo[:,1])]) + np.array([image_lx+10,image_ly+10 ]) #ten pixels margin
    final_image = np.zeros([list_image[0].shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])] )

    for ind_img in range(len(list_cordo)):
        cx, cy = list_cordo[ind_img]
        cx = round(cx)
        cy = round(cy)
        final_image[:, cy : cy + image_ly, cx:cx +  image_lx] = list_image[ind_img]

    input = np.amax(final_image, 0)
    pa_ch1, pb_ch1 = np.percentile(input, (1,99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')

    fig, ax = plt.subplots(1, 1, figsize=(40, 40))
    ax.imshow(final_image_mip)
    plt.show()
    return final_image

#%%
def spots_detection_stitching(list_spot, list_cordo, image_shape = [55, 2048, 2048]):







    list_cordo = np.array(list_cordo).round()
    list_cordo = list_cordo - np.array([min([x[0] for x in list_cordo]),
                                       min([x[1] for x in list_cordo])])

    print(list_cordo)
    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([np.max(list_cordo[:, 0]),np.max(list_cordo[:,1])]) + np.array([image_lx+100,image_ly+100 ]) #ten pixels margin
    masks = np.zeros([image_shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])] )
    print(masks.shape)
    new_spot_list = []

    for ind_img in range(len(list_cordo)):
        spot_list = list_spot[ind_img]
        cx, cy = list_cordo[ind_img]
        cx = int(cx)
        cy  = int(cy)
        print(list_cordo[ind_img])
        print(len(new_spot_list) + len(spot_list))
        print(len(spot_list))
        for sp in spot_list:

            if masks[int(sp[0]), int(sp[1] + cy), int(sp[2] + cx)] == 0:
                    new_spot_list.append(sp +np.array([0, cy, cx]))
                   # print(sp +np.array([0, cy, cx]))
                   # print(([0, cy, cx]))
        masks[:, cy : cy + image_ly, cx:cx +  image_lx] = np.ones([image_shape[0],image_ly, image_lx ])
        print(len(new_spot_list))
        print()
    return new_spot_list





#%%

if __name__  == "__main__":
#%%
import numpy as np
import tifffile


### no need to  stitch the dapi mask

################
#stich the point
#################

#### first stich all round together with list cordo from imageJ

dico_list_cordo = {"r1_bc1" : [(0.0, 0.0),
                                (1852.7813590145313, 37.89905459680426),
                                (3701.5989729918037, 73.44660408534749,),
                                (-34.709682598870074, 1854.6090833798212),
                                (1817.9228871430223, 1894.4869399371014),
                                (3667.095811040399, 1927.9497196796233),
                                (-69.54245415267906, 3711.0475146347926),
                                (1784.1259592068416, 3747.074571350263),
                                (3632.047378843126, 3783.831448610312)],

                    "r3_bc4" :[(0.0, 0.0),
                        (1854.1616703182967, 41.96625436587307),
                        (3703.143372711853, 77.31952835427283),
                        (-33.92455191285654, 1857.3443243553254),
                        (1819.7631471038144, 1898.677290912805),
                        (3669.5216709089545, 1932.2470286380164),
                        (-68.6808815397323, 3714.769595088861),
                        (1785.9035016466291, 3751.614828825737),
                        (3634.0560832634083, 3788.8247795809384)],

                    "r4_bc5": [(0.0, 0.0),
                                   (1856.0394935014735, 39.45104361608831),
                                    (3703.931721437577, 74.77719467771846),
                                     (-33.69012082420835, 1856.3004734323827),
                                     (1821.637365648822, 1896.2862289782827),
                                     (3670.1474987639394, 1929.5443106840899),
                                     (-68.46131615251147, 3713.4765144846915),
                                        (1787.3309888146837, 3750.1389471890625),
                                        (3635.024743439781, 3786.286507792005) ],
                    "r5_bc6":
                                    [(0.0, 0.0),
                                    (1856.6775838930862, 41.24406517256911),
                                    (3704.386805332943, 76.10862215386476),
                                    (-34.05178049442486, 1856.4763467785983),
                                    (1821.3749307565452, 1897.2551016160578),
                                    (3670.1609772584043, 1930.766499427954),
                                    (-68.9800251445539, 3714.611958002779),
                                     (1787.5441151245368, 3751.293280127653),
                                    (3635.350664577769, 3787.425807994482)],

                    "r6_bc7":[(0.0, 0.0),
                             (1855.031187682188, 36.89593447719009),
                            (3703.2444285681604, 72.36402642261444),
                            (-33.514930013352, 1854.8862381286754),
                            (1820.4581977744158, 1893.1712695798183),
                            (3669.46596959303, 1925.850338270596),
                            (-68.60486120758135, 3711.5803654834344),
                            (1786.3112862349196, 3747.2951674073697),
                            (3633.8971476871207, 3782.5620922397866)],

                    "r7_bc3":[(0.0, 0.0),
                                      (1852.724960164594, 39.219440139148446),
                                    (3699.974645564619, 72.28774786417804),
                                        (-33.73006758264091, 1852.7660650677128),
                                        (1816.227249895914, 1892.3515436720618),
                                    (3669.46596959303, 1925.850338270596),
                                      (-68.73006758264091, 3705.7660650677126),
                                    (1769.5866467242683, 3659.9590577960057),
                                    (3615.9067050561034, 3697.155080336764)]
                            }



## generate list dico


dico_spots = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition1/new_dico_spots.npy', allow_pickle=True).item()
dico_spots1103 = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/sp/dico_spots1103.npy', allow_pickle=True).item()
dico_spots_r7 = np.load("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/sp/dico_beadr7.npy", allow_pickle=True).item()


dico_spots['r6_bc7'] = dico_spots1103['r6_bc7']
dico_spots['r7_bc3'] = dico_spots_r7['r7_bc3']


np.save('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots1103', dico_spots)

dico_spots = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots1103.npy', allow_pickle=True).item()
dico_list_spot = {}
for round in dico_spots:
    dico_list_spot[round] = []
    print()
    print(round)
    for posi in range(9):
        kk = f'opool1_1_MMStack_3-Pos_{posi}_ch1.tif'
        if round == 'r6_bc7' or round == 'r7_bc3':
            dico_list_spot[round].append(dico_spots[round][kk])
            print(len(dico_spots[round][kk]))
        else:
            dico_list_spot[round].append(dico_spots[round][kk][0])
            print(len(dico_spots[round][kk][0]))
##%
dico_spots_stitch = {}
for round in dico_list_spot:
    st_spots = spots_detection_stitching(list_spot = dico_list_spot[round], list_cordo=dico_list_cordo[round])
    dico_spots_stitch[round] = st_spots

np.save('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots_stitch', dico_spots_stitch)


#%%%
####
# register point cloud to first round
####
def compute_the_translation(static_image, moving_image,
                            upsample_factor=100,
                            return_translated_image = False):
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    shift, error, diffphase = phase_cross_correlation(static_image, moving_image,  upsample_factor=upsample_factor)
    # moving_image + shift = static_image

    if return_translated_image:
        offset_image = fourier_shift(np.fft.fftn(moving_image), shift)
        offset_image = np.fft.ifftn(offset_image).real
        return shift, error, diffphase, offset_image

    return shift, error, diffphase


path_to_fused = Path("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/fish_stich")
list_fused = list(path_to_fused.glob("*r*"))

dico_shift = {}
static_image = np.amax(tifffile.imread("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/fish_stich/Fused_ch1_r1_bc1.tif"), 1)
for path_moving in list_fused[1:]:
    print(path_moving)
    moving_image = np.amax(tifffile.imread(str(path_moving)), 1)
    if str(path_moving)[-10:-4] != "r7_bc5":
        #moving_image = moving_image[200:5000, 200:5000]
        moving_image = moving_image[:, 200:5000]
        shift, error, diffphase = compute_the_translation(static_image[:, 200:5000],
                                                          moving_image, return_translated_image=False)
    else:
        #moving_image = moving_image[:3500, 3500]
        moving_image = moving_image[:, 200:5000]
        shift, error, diffphase = compute_the_translation(static_image[:, 200:5000],
                                                          moving_image, return_translated_image=False)
    print(shift)
    dico_shift[str(path_moving)[-10:-4]]  = shift
np.save("dico_shift", dico_shift)


#############"
#%%cluster point cloud
##################
dico_param_probes = {"Lamp3": (32, 0.42),
                     "Pdgfra": (35, 0.42),
                     "Chil3": (30, 0.35),
                     'Cap': (35, 0.30),
                     'aCap': (35, 0.30),
                     "Ptprb": (27, 0.45),
                     "Fibin": (27, 0.40),
                     'C3ar1': (35, 0.45),
                     'Hhip': (35, 0.25),
                     'Mki67': (40, 0.30),
                     "Serpine1": (40, 0.50),
                     "Apln": (30, 0.40),
                     "Pecam1": (30, 0.40),
                     "CEC": (35, 0.30),
                     "Rtkn2": (34, 0.30),
                     }


dico_bc_gene = {
    'r1_bc1' : "Rtkn2",
    'r3_bc4' : "Pecam1",
    'r4_bc5': "Ptprb",
    'r5_bc6':  "Pdgfra",
    'r6_bc7': "Chil3",
    'r7_bc3': "Lamp3"}


from spots.spot_detection import computer_optics_cluster, cluster_over_nuclei_3D_convex_hull

img_dapi_mask = tifffile.imread("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/fish_stich/Fused_ch1_r7_bc3.tif")
dico_res_classif = np.load("dico_res_classif.npy", allow_pickle=True).item()

dico_spots_stitch = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots_stitch.npy', allow_pickle=True).item()
dico_shift = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_shift.npy', allow_pickle=True).item()
dico_res_classif_chil3 = {}
for round in ['r6_bc7']:
    spots = np.array(dico_spots_stitch[round])

    if round in list(dico_shift.keys()):
        spots = spots + np.array([0, int(dico_shift[round][0]), int(dico_shift[round][1]) ])

    eps, iou_threshold =  dico_param_probes[dico_bc_gene[round]]


    labels = computer_optics_cluster(spots, eps=eps, min_samples=4,
                                           min_cluster_size=4, xi=0.05, scale=np.array([(270 / 107), 1, 1]))
    #_, _, _, labels =  dico_res_classif[round]


    nuclei_positive, positive_cluster_568, negative_cluster_568 = cluster_over_nuclei_3D_convex_hull(
        labels,
        spots,
        img_dapi_mask,
        iou_threshold=iou_threshold,
        scale=[270, 107, 107])
    dico_res_classif_chil3[round] = [nuclei_positive, positive_cluster_568, negative_cluster_568, labels]
    print("number of positive")/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition2/r1_bc1/mask/_80ft0.75/dapi_maskdapi.tif
    print(len(nuclei_positive))
    print(round)
    print()

np.save("dico_res_classif_chil3", dico_res_classif_chil3)


import


#%%
import napari
import tifffile
import numpy as np

dico_spots_stitch = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots_stitch.npy', allow_pickle=True).item()

fish = tifffile.imread("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_2_ch1.tif")
dapi = tifffile.imread("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition2/r1_bc1/opool1_1_MMStack_3-Pos_2_ch3.tif")
dico_spots = np.load("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots1103.npy", allow_pickle=True).item()
size = 4
viewer = napari.Viewer()

st_spots = dico_spots["r7_bc3"]["opool1_1_MMStack_3-Pos_2_ch1.tif"]
viewer.add_points(st_spots, size=size,
                  edge_color="green", face_color="green", name="r7_bc3")# ndim=3, scale = [0.3, 0.1, 0.1])
#ch1 = tifffile.imread("/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_1_ch1.tif")
viewer.add_image(np.amax(fish, 0),
                 name='fishr7_bc3',
                 gamma=0.5 )#, scale = [0.3, 0.1, 0.1])

viewer.add_image(np.amax(dapi, 0),
                 name='dapi',
                 gamma=0.5)#, scale = [0.3, 0.1, 0.1])


# Define the number of dimensions we are working on
dim = 3

#### then use the translation from the phase correlation to alligned each round




## ## sort round 6




#%% clustering

    #                        "Lamp3": (32, 0.42),
    import random
    from spots.spot_detection import computer_optics_cluster,cluster_over_nuclei_3D_convex_hull

    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    spots = np.load('/home/tom/Bureau/stiched_images_florian/path_raw/fish/stitch_detected_spots.npy')
    img_dapi_mask = tifffile.imread("/home/tom/Bureau/stiched_images_florian/path_raw/to_segment/res4/_80ft0.75/dapi_mask106z_ignoredapi.tif")
    labels_lamps = computer_optics_cluster(spots, eps=32, min_samples=4,
                                         min_cluster_size=4, xi=0.05, scale = np.array([(270/107),1,1]))


   # def cluster_over_nuclei_3D_convex_hull(labels, spots, masks, iou_threshold=0.5, scale=[300, 103, 103]):

    nuclei_positive, positive_cluster_568, negative_cluster_568 = cluster_over_nuclei_3D_convex_hull(
        labels_lamps,
        spots,
        img_dapi_mask,
        iou_threshold=0.42,
        scale= [270, 107, 107] )

    nuclei_568_1 = nuclei_positive
    nuclei_647_1 = []


    ####
    # PLOT##########################
    # plot final classification
    ######
    from spots.spot_detection import mask_image_to_rgb2D_from_list_green_cy3_red_cy5_both_blue_grey

    colors = np.zeros((4, 1))
    colors[0, 0] = 0.6  # blue cy3
    colors[1, 0] = 0.01  # red cy5
    colors[2, 0] = 0.5  # grey
    colors[3, 0] = 0.6  # Blur
    fig, ax = plt.subplots(1, 1, figsize=(30, 20))


    m, green, yellow, blue, purple = mask_image_to_rgb2D_from_list_green_cy3_red_cy5_both_blue_grey(
        mip_dapi,
        np.amax(img_dapi_mask, 0), nuclei_568_1, nuclei_647_1, colors)
    ax.imshow(m)
    plt.show()


#%%
    #
    import random

    dico_bc_gene = {
        'r1_bc1' : "Rtkn2",
        'r3_bc4' : "Pecam1",
        'r4_bc5': "Ptprb",
        'r5_bc6':  "Pdgfra",
        'r6_bc7': "Chil3",
        'r7_bc3': "Lamp3"}


    dico_color_hex = {
        "Rtkn2": "#FF0000",
        "Pecam1": "#FF9A00",
        "Ptprb": "#E2FF00",
        "Pdgfra": "#00FFEF",
        "Chil3": "#002BFF",
        "Lamp3": "#FF00E6",
}
    dico_res_classif = np.load("dico_res_classif.npy", allow_pickle = True).item()
    dico_res_classif2 = np.load("dico_res_classif2.npy", allow_pickle=True).item()
    dico_res_classif_Rtkn2 = np.load("dico_res_classif_Rtkn2.npy", allow_pickle=True).item()
    dico_res_classif_chil3 = np.load("dico_res_classif_chil3.npy", allow_pickle=True).item()

    dico_spots_stitch = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_spots_stitch.npy',
                                allow_pickle=True).item()
    dico_shift = np.load('/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/dico_shift.npy',
                         allow_pickle=True).item()

    nuclei_positive, positive_cluster_568, negative_cluster_568, labels = dico_res_classif_Rtkn2["r1_bc1"]
    spots = dico_spots_stitch["r1_bc1"]
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    list_color = get_colors(5000)


    mip_mask = tifffile.imread("mip_mask.tif")


    mip_dapi = tifffile.imread("dapi.tiff")
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))




    ax.imshow(mip_dapi, cmap = "gray")
    for k in dico_res_classif2:
        if k != 'r1_bc1' and k!= 'r6_bc7':
            spots = dico_spots_stitch[k]
            positive_cell, positive_cluster_568, negative_cluster_568, labels = dico_res_classif2[k]
            set_cluster_568 = [el[0] for el in positive_cluster_568]  # np.unique(labels)#[el[0] for el in positive_cluster_568] + [el[0] for el in negative_cluster_568]
            for s_index in range(len(spots)):
                if labels[s_index] in set_cluster_568:
                    s = spots[s_index]
                    ax.scatter(s[-1], s[-2], c=dico_color_hex[dico_bc_gene[k]], s=3)

    for k in dico_res_classif_Rtkn2:
            spots = dico_spots_stitch[k]

            positive_cell,positive_cluster_568, negative_cluster_568, labels  = dico_res_classif_Rtkn2[k]
            set_cluster_568 = [el[0] for el in positive_cluster_568]
            for s_index in range(len(spots)):
                if labels[s_index] in set_cluster_568:
                    s = spots[s_index]
                    ax.scatter(s[-1], s[-2], c=dico_color_hex[dico_bc_gene[k]], s=3)


    for k in dico_res_classif_chil3:
            spots = dico_spots_stitch[k]

            positive_cell,positive_cluster_568, negative_cluster_568, labels  = dico_res_classif_chil3[k]
            set_cluster_568 = [el[0] for el in positive_cluster_568]
            for s_index in range(len(spots)):
                if labels[s_index] in set_cluster_568:
                    s = spots[s_index]
                    ax.scatter(s[-1], s[-2], c=dico_color_hex[dico_bc_gene[k]], s=3)


    plt.show()


        if labels[s_index] == -1:
            s = spots[s_index]
            ax.scatter(s[-1], s[-2], c="orange", s=3)


    from scipy.spatial import ConvexHull
    for c in set_cluster_568:
        point_cloud = []
        for s_index in range(len(spots)):
            if labels[s_index] == c:
                point_cloud.append([spots[s_index][2], spots[s_index][1]])
        points = np.array(point_cloud)
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'c')
            ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1,
                    markersize=5)


#%% renaming scrip
#

   def rescale_point(zxy, offset_x, offset_y, limit_x, limit_y):
        if zxy[-1] > limit_x or  zxy[-2] > limit_y or zxy[-1] < offset_x or  zxy[-2] < offset_y :
            return None
        else:

            zxy = np.array([zxy[0], zxy[1] - offset_x, zxy[2] - offset_y])

from pathlib import Path
import os

path_to_data = "/media/tom/Transcend/data020322/2022-02-24_opool-1/multi_channel_stacks"

path = Path(path_to_data)


list_path_round = list(path.glob("*"))


for p in list_path_round:
    list_image = list(p.glob("*.tif"))
    for image_path in list_image:
        image_name = str(image_path)
        x_index = int(image_name[-11:-8])
        y_index = int(image_name[-15:-12])
        index_pos = x_index + y_index*3
        os.rename(image_name, image_name.replace(image_name[-15:-8], str(index_pos)))


#%%

dico_res_classif2 = np.load("dico_res_classif2.npy", allow_pickle=True).item()
dico_res_classif_Rtkn2 = np.load("dico_res_classif_Rtkn2.npy", allow_pickle=True).item()
dico_res_classif_chil3 = np.load("dico_res_classif_chil3.npy", allow_pickle=True).item()
mip_mask = tifffile.imread("mip_mask.tif")
mip_dapi = tifffile.imread("dapi.tiff")
dico_color = {
    "Rtkn2": 0.00, #FF0000
    "Pecam1": 0.10, #FF9A00
    "Ptprb": 0.18, #E2FF00
    "Pdgfra": 0.49, #00FFEF
    "Chil3": 0.64, #002BFF
    "Lamp3": 0.85, #FF00E6
}
## generate mask dico
masks_dico = {}
list_pos = []
for k in dico_res_classif2:
    if k != 'r1_bc1' and k!= 'r6_bc7':
        positive_cell,_,_,_ = dico_res_classif2[k]
        for cell  in positive_cell:
            masks_dico[cell] = dico_color[dico_bc_gene[k]]

for k in dico_res_classif_Rtkn2:
        positive_cell,_,_,_ = dico_res_classif_Rtkn2[k]
        for cell  in positive_cell:
            masks_dico[cell] = dico_color[dico_bc_gene[k]]


for k in dico_res_classif_chil3:
        positive_cell,_,_,_ = dico_res_classif_chil3[k]
        for cell  in positive_cell:
            masks_dico[cell] = dico_color[dico_bc_gene[k]]


def mask_image_to_rgb2D(img,masks, masks_dico):
    if img.ndim>2:
        img = np.amax(img, 0).astype(np.float32)
    else:
        img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    green = 0
    yellow = 0
    purple = 0
    blue = 0

    for n in np.unique(masks):
        if n==0:
            continue
        ipix = (masks==n).nonzero()
        if n in list(masks_dico.keys()):
            HSV[ipix[0],ipix[1],0] =masks_dico[n]
            yellow += 1
            HSV[ipix[0],ipix[1],1] = 1

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8) #
    return RGB, green, yellow, blue, purple #green norna, yellow cy3, purle #both  blue #cy5


    fig, ax = plt.subplots(1, 1, figsize=(30, 20))


    m, green, yellow, blue, purple =  mask_image_to_rgb2D(mip_dapi, mip_mask, masks_dico)
    ax.imshow(m)
    plt.show()


def mask_image_to_rgb2D(img,masks, masks_dico):
    if img.ndim>2:
        img = np.amax(img, 0).astype(np.float32)
    else:
        img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    green = 0
    yellow = 0
    purple = 0
    blue = 0
    for n in np.unique(masks):
        if n==0:
            continue
        ipix = (masks==n).nonzero()
        HSV[ipix[0],ipix[1],0] = 0.55
        yellow += 1
        HSV[ipix[0],ipix[1],1] = 1

    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8) #
    return RGB, green, yellow, blue, purple #green norna, yellow cy3, purle #both  blue #cy5


fig, ax = plt.subplots(1, 1, figsize=(30, 20))


m, green, yellow, blue, purple =  mask_image_to_rgb2D(mip_dapi, mip_mask, masks_dico)
ax.imshow(m)
plt.show()