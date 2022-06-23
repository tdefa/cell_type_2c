
import tifffile
from matplotlib import pyplot as plt
import numpy as np


from pathlib import Path

path_dapi  = "/home/tom/Bureau/stiched_images_florian/dapi/"

p = Path("/home/tom/Bureau/stiched_images_florian/dapi/")

for n_p in list(p.glob("*.tiff")):
    m = tifffile.imread(str(n_p))
    plt.imshow(np.amax(m, 0))
    plt.title(str(n_p).split('/')[-1])
    plt.show()


p = Path("/home/tom/Bureau/stiched_images_florian/dapi/")

for n_p in list(p.glob("*.tiff")):
    m = tifffile.imread(str(n_p))
    plt.imshow(np.amax(m, 0))
    plt.title(str(n_p).split('/')[-1])
    plt.show()


#%%

import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import tifffile
import numpy as np
#from cellpose import models, io, plot

from skimage.exposure import rescale_intensity

from pathlib import Path
import argparse


img = tifffile.imread("/home/tom/Downloads/lamp3_5_MMStack_3-Pos_002_002.ome(4).tif")

input = np.amax(img[:,1], 0)
pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(final_image_mip)
plt.show()


p = Path("/home/tom/Bureau/stiched_images_florian/dapi/")

for n_p in list(p.glob("*.tiff")):
    m = tifffile.imread(str(n_p))
    plt.imshow(np.amax(m, 0))
    plt.title(str(n_p).split('/')[-1])
    plt.show()



for inn in range(25, 28):
    input = img[inn, 0]
    pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    plt.imshow(final_image_mip)
    plt.title(str(inn) + " 0")
    plt.show()

    input = img[inn, 1]
    pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    plt.imshow(final_image_mip)
    plt.title(str(inn) + " 1")
    plt.show()

### create a mip
figsize = (50, 50)
input = tifffile.imread("/home/tom/Bureau/stiched_images_florian/path_raw/mip_fish/stitch.tif")
pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
fig, ax = plt.subplots(1, 1, figsize=figsize)
ax.imshow(final_image_mip)
plt.show()


#%%

from pathlib import Path

path = Path("/home/tom/Bureau/stiched_images_florian/old_folder/path_raw/dapi/")
list_img = np.sort(list(path.glob("*.tiff")))
for path_image in range(len(list_img)):
    m = tifffile.imread(str(list_img[path_image]))
    tifffile.imsave("/home/tom/Bureau/stiched_images_florian/old_folder/path_raw/mip_dapi/tile_"+str(path_image),  np.amax(m,0))
    input = np.amax(m,0)
    pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    plt.imshow(final_image_mip)
    plt.title(str(path_image).split('/')[-1])
    plt.show()



#

path_raw_tiff = "/home/tom/Bureau/stiched_images_florian/path_raw/"


p = Path(path_raw_tiff + "ome_tiff")
for lt in p.glob("*f"):
    m = tifffile.imread(str(lt))

    double_image  = m.reshape(106, 2048, 2048)
    fish = double_image[:53]
    dapi = double_image[53:]

    input = np.amax(fish, 0)
    pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    plt.imshow(final_image_mip)
    plt.title("fish")
    plt.show()

    input = np.amax(dapi, 0)
    pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    plt.imshow(final_image_mip)
    plt.title("dapi")
    plt.show()

    name_image = str(lt).split('/')[-1][:-8]

    tifffile.imsave(path_raw_tiff + "fish/" + name_image + ".tiff", fish)
    tifffile.imsave(path_raw_tiff + "dapi/" + name_image + ".tiff", dapi)

#%% transform to mip

path = Path("/home/tom/Bureau/stiched_images_florian/mip_fish_dapi/ome_tiff1065353/")
list_img = np.sort(list(path.glob("*.tiff")))
for path_image in range(len(list_img)):
    m = tifffile.imread(str(list_img[path_image]))
    print(m.shape)
    tifffile.imsave("/home/tom/Bureau/stiched_images_florian/mip_fish_dapi/mip_fish_dapi/tile_"+str(path_image)+'.tif',  np.amax(m,0))
    input = np.amax(m,0)
    pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
    final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
    plt.imshow(final_image_mip)
    plt.title(str(path_image).split('/')[-1])
    plt.show()



#%%
import napari
fish = tifffile.imread("/home/tom/Bureau/stiched_images_florian/path_raw/ome_tiff1065353/smfish/106z_ignorefish.tif")
cy5_point = np.load("/home/tom/Bureau/stiched_images_florian/path_raw/fish/stitch_detected_spots.npy")

viewer = napari.Viewer()
viewer.add_image(fish, name = 'mask', scale=(1, 0.33, 0.33))

viewer.add_points(cy5_point, size=3, scale=(3, 0.33, 0.33), edge_color = "red",face_color = "red",   ndim=3)

mip_fish = np.amax(fish, 0)
mip_spot = [[i[1], i[2]] for i in cy5_point]

viewer = napari.Viewer()
viewer.add_image(mip_fish, name = 'mask')

viewer.add_points(mip_spot, size=3, edge_color = "red",face_color = "red",  )


dapi = tifffile.imread("/home/tom/Bureau/stiched_images_florian/path_raw/to_segment/106z_ignoredapi.tif")
mask = tifffile.imread("/home/tom/Bureau/stiched_images_florian/path_raw/to_segment/res4/_80ft0.75/dapi_mask106z_ignoredapi.tif")
input = np.amax(dapi, 0)
pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
mip_dapi = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
mip_mask = np.amax(mask, 0)


fig, ax = plt.subplots(1, 1, sharex='col', figsize=(40, 40))
ax.imshow(mip_dapi, alpha = 0.75,  cmap="gray")
#ax.imshow(mip_mask > 0, alpha = 0.25, cmap="Reds")
plt.title("80")
plt.show()



fig, ax = plt.subplots(1, 1, sharex='col', figsize=(40, 40))

input = np.amax(fish, 0)
pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
ax.imshow(final_image_mip)

from matplotlib.patches import RegularPolygon

for xyz in cy5_point:
    ##define color
    x = RegularPolygon((xyz[-1], xyz[-2]), 5, radius =5, color=colors[9], linewidth=1,
                       fill=True)  # or usefct from from matplotlib.patches import RegularPolygon
    ax.add_patch(x)
plt.show()

#%% plot clustering
#labels_lamps = computer_optics_cluster(spots, eps=32, min_samples=4,
#                                       min_cluster_size=4, xi=0.05, scale=np.array([(270 / 107), 1, 1]))

fig, ax = plt.subplots(1, 1, sharex='col', figsize=(40, 40))

input = np.amax(fish, 0)
pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
ax.imshow(final_image_mip, cmap = 'gray')
cmap = plt.get_cmap('RdYlBu')
colors = np.random.rand(len(np.unique(labels_lamps)), 3)
from matplotlib.patches import RegularPolygon

for i in range(len(labels_lamps)):
    if labels_lamps[i] == -1:
        continue
    xyz = cy5_point[i]
    x = RegularPolygon((xyz[-1], xyz[-2]), 7, radius =5, color=colors[labels_lamps[i]], linewidth=1,
                       fill=True)  # or usefct from from matplotlib.patches import RegularPolygon
    ax.add_patch(x)
plt.show()



fig, ax = plt.subplots(1, 1, sharex='col', figsize=(40, 40))

input = np.amax(fish, 0)
pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
final_image_mip = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
ax.imshow(final_image_mip, cmap = 'gray')
cmap = plt.get_cmap('RdYlBu')
colors = np.random.rand(len(np.unique(labels_lamps)), 3)
from matplotlib.patches import RegularPolygon

for i in range(len(labels_lamps)):
    xyz = cy5_point[i]
    x = RegularPolygon((xyz[-1], xyz[-2]), 7, radius =5, color="red", linewidth=1,
                       fill=True)  # or usefct from from matplotlib.patches import RegularPolygon
    ax.add_patch(x)
plt.show()

#%%

p = Path(path_raw_tiff )
for lt in p.glob("*f"):
    m = tifffile.imread(str(lt))

    double_image  = m.reshape(106, 2048, 2048)


    name_image = str(lt).split('/')[-1][:-8]
    tifffile.imsave("/home/tom/Bureau/stiched_images_florian/path_raw/ome_tiff1065353/" + name_image + ".tiff", double_image)

####################"""
for i in range(9):
    try:
        r = tifffile.imread(f'/media/tom/Transcend/data020322/2022-02-24_opool-1/multi_channel_stacks/r7_bc3/opool1_1_MMStack_3-Pos_{str(i)}.ome.tif')
        if r.shape == (55, 2, 2048, 2048):

            tifffile.imsave(f'/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_{str(i)}_ch1.tif', r[:,0])
            tifffile.imsave(f'/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_{str(i)}_ch2.tif', r[:,1])

    except :
        z = np.zeros([55, 2048, 2048])
        tifffile.imsave(f'/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_{str(i)}_ch1.tif', z)
        tifffile.imsave(f'/media/tom/Transcend/data020322/2022-02-24_opool-1/acquisition/r7_bc3/opool1_1_MMStack_3-Pos_{str(i)}_ch2.tif', z)




if __name__ == "__main__": # code used for the first stiching round
    path_to_dapi = "/home/tom/Bureau/stiched_images_florian/path_raw/fish/smFish/"
    p = Path(path_to_dapi)
    list_path = np.sort(list(p.glob("lamp*.tiff")))
    list_image = []
    for pda in list_path:
        list_image.append(tifffile.imread(str(pda)))

    list_cordo =[(0.0, 0.0),
                (1944.9088035984357, 66.4747988495096),
                (3886.5507930299887, 132.4035318599818),
                (-65.91005455373279, 1940.517996883836),
                (1879.1787495948329, 2010.0294210208194),
                (3820.1954803897916, 2072.332801930645),
                (-121.91005455373279, 3906.5179968838356),
                (1813.5161342925915, 3951.7982599087527),
                (3754.855293958105, 4018.5658826671433)]

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
    ax.imshow(np.amax(final_image, 0))
    plt.show()



#%%  spots detection stiching
    from spots.spot_detection import spot_detection_for_stiching

    image_shape = [53, 2048, 2048]

    path_to_dapi = "/home/tom/Bureau/stiched_images_florian/path_raw/fish/spots_detection/"
    p = Path(path_to_dapi)
    list_path = np.sort(list(p.glob("lamp*.npy")))
    list_spot = []
    for pda in list_path:
        list_spot.append(np.load(str(pda)))


    list_cordo =[(0.0, 0.0),
                (1943.7495524499063, 66.08346372173153),
                (3884.8750383481847, 132.49977883232978),
                (-65.75184265950702, 1938.9172995095473),
                (1878.3748438451623, 2008.7494611558518),
                (3820.0001822883073, 2071.9149594541154),
                (-131.6290760740615, 3883.002049848519),
                (1812.499029950186, 3950.0904862006178),
                (3753.7520901238613, 4016.49689622804)]

    list_cordo = np.array(list_cordo).round()
    list_cordo = list_cordo - np.array([min([x[0] for x in list_cordo]),
                                       min([x[1] for x in list_cordo])])
    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([np.max(list_cordo[:, 0]),np.max(list_cordo[:,1])]) + np.array([image_lx+10,image_ly+10 ]) #ten pixels margin
    masks = np.zeros([image_shape[0], int(final_shape_xy[0]), int(final_shape_xy[1])] )
    new_spot_list = []

    for ind_img in range(len(list_cordo)):
        spot_list = list_spot[ind_img]
        cx, cy = list_cordo[ind_img]
        cx = round(cx)
        cy = round(cy)
        print(len(new_spot_list) + len(spot_list))
        print(len(spot_list))
        for sp in spot_list:
            if masks[sp[0], sp[1] + cy, sp[2] + cx] == 0:
                    new_spot_list.append(sp +np.array([0, cy, cx]))
        masks[:, cy : cy + image_ly, cx:cx +  image_lx] = np.ones([53,image_ly, image_lx ])
        print(len(new_spot_list))
        print()


    np.save('/home/tom/Bureau/stiched_images_florian/path_raw/fish/', new_spot_list)