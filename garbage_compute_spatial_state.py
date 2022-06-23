
def generate_exels_one_cell(list_param):
    """

    Parameters
    ----------
    list_param

    Returns
    -------

    """
    list_folder, gene_smfish, path_to_take, path_save =  list_param[0], list_param[1], list_param[2], list_param[3]
    try:

if __name__ == '__main__':


    path_to_take = "/home/tom/Bureau/210205_Prolicence/"

list_folder = [
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/",
    # ok spot ok image, on image is wrong
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/",  # pb to rerun
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/",  # ok spot
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/",  ##pb
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/",
    "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/",
]

# %%
for probes in list_probes:
    update_dataframecell_type([dico_param_probes,
                               list_folder, ['Lamp3'], "/home/tom/Bureau/annotation/exels_folders/one_cells_analysis/"])
    print("ok")

    list_param = [dico_param_probes,
                  list_folder, probes, "/home/tom/Bureau/annotation/exels_folders/one_cells_analysis/"]

# %% distributed version cell type

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
]
import multiprocessing

l_params = []
l_params = []
list_probes = [['Lamp3'], ['Pdgfra'], ['Serpine1'], ['Ptprb'], ['Apln'], ['Chil3'], ['CEC'], ['Fibin'], ['C3ar1'],
               ['Hhip'], ['Mki67'],
               ['Pecam1'], ['Cap', 'aCap'],
               ]  #
for prb in list_probes:
    l_params.append([
        list_folder,
        prb,
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/one_cells_analysis_sp/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/one_cells_analysis_sp22/", ])

# %%
number_processes = 6
pool = multiprocessing.Pool(number_processes)
results = pool.map_async(cell_type_point_cloud_update, l_params)
pool.close()
pool.join()

# %%

list_probes_type = [['Pecam1'], ['CEC'], ['Lamp3'], ['Pdgfra'], ['Chil3'], ['Cap', 'aCap'], ['Ptprb'],
                    ['Fibin'], ['C3ar1'], ['Hhip'], ['Apln'],
                    ]
for gene_type in list_probes_type:
    generate_exels_cell_state_type([dico_param_probes,
                                    list_folder, gene_type, ['Serpine1'],
                                    "/home/tom/Bureau/annotation/exels_folder/cell_state_cell_type/"])

# %% distributed version cell type cell state
import multiprocessing

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
]
l_params = []
list_probes_type = [['Lamp3'], ['Pdgfra'], ['Chil3'], ['Cap', 'aCap'], ['Ptprb'],
                    ['Fibin'], ['C3ar1'], ['Hhip'], ['Apln'],
                    ['Pecam1'], ['CEC']]
for gene_type in list_probes_type:
    l_params.append([
        list_folder,
        gene_type, ['Serpine1'],
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/cell_state_cell_type/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/cell_state_cell_type22/"])
for gene_type in list_probes_type:
    l_params.append([
        list_folder,
        gene_type, ['Mki67'],
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/cell_state_cell_type/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/cell_state_cell_type22/"])
    # %%
number_processes = 6
pool = multiprocessing.Pool(number_processes)
results = pool.map_async(double_cells_point_cloud_update, l_params)
pool.close()

pool.join()

"""import matplotlib.pyplot as plt
from skimage import draw

#from skimage.draw import disk
shape = (400, 400)
img = np.zeros(shape, dtype=np.uint8)
rr, cc = disk((100, 100), 50, shape=shape)
img[rr, cc] = 1
plt.imshow(img)
plt.show()

img = np.stack([img]*50)"""

# %% distributed version celll type type
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
]
l_params = []
list_couple_probes_type = [[['Lamp3'], ['Pdgfra']], [['C3ar1'], ['Chil3']], [['Pdgfra'], ['Hhip']],
                           [['Pecam1'], ['Apln']],
                           [['Pecam1'], ['Ptprb']]]  # ,
for gene_type in list_couple_probes_type:
    l_params.append([
        list_folder,
        gene_type[0], gene_type[1],
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/cell_type_couple_sp/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/exels_folders/exels_from_thalassa/cell_type_couple_sp2206/"])
# %%
number_processes = 3
pool = multiprocessing.Pool(number_processes)
results = pool.map_async(double_cells_point_cloud_update, l_params)
pool.close()
pool.join()

# %%

if __name__ == '__main__':

    list_folder = [
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Capillary_EC/",
        # ok spot ok image, on image is wrong
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Large_Vessels/",  # pb to rerun
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200828-NIvsIR5M/00_Macrophages/",  # ok spot
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_CEC/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/200908_fibrosis/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/201030_fridyay/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/201127_AM_fibro/",  ##pb
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_prolif/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210205_Prolicence/aCap_senes/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210219_myo_fibros_y_macrophages/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/IR5M/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210412_repeat_fibro/NI/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210413_rep2/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210425_angiogenesis/",
        "/home/tom/Bureau/annotation/cell_type_annotation/to_take/210426_repeat3/",
    ]

    dico_param_probes = {"Lamp3": (32, 0.42),
                         "Pdgfra": (35, 0.42),
                         "Chil3": (15, 0.55),
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
                         "CEC": (0, 2),
                         }

    list_probes = [
        ['Lamp3'], ['Pdgfra'],
        ['Chil3'], ['Cap', 'aCap'], ['Ptprb'],
        ['Fibin'], ['C3ar1'], ['Hhip'], ['Mki67'], ['Serpine1'], ['Apln'],
        ['Pecam1'], ['CEC']]

    # dico_param_probes, list_folder, gene_smfish, path_save = list_param[0], list_param[1], list_param[2], list_param[3]
    gene_smfish = list_probes[0]
    for folder_name in list_folder:
        path_output_segmentaton = folder_name + "tiff_data/" + "predicted_mask_dapi/"
        onlyfiles = [f for f in listdir(path_output_segmentaton) if
                     isfile(join(path_output_segmentaton, f)) and f[-1] == "f"]
        onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]

        if os.path.isfile(folder_name + "dico_stat_1305.npy"):
            dico_stat = np.load(folder_name + "dico_stat_1305.npy", allow_pickle=True).item()
            print("dictionary choice dico_stat_1305" + str(Path(folder_name).parts[-2:]))
        else:
            dico_stat = np.load(folder_name + "dico_stat.npy", allow_pickle=True).item()
            print("dictionary choice dico_stat" + str(Path(folder_name).parts[-2:]))
        sorted_name = np.sort(list(dico_stat.keys()))
        for key_cell_name in sorted_name:
            if not any(word in key_cell_name for word in gene_smfish):
                continue
            t = time.time()

            print(key_cell_name)

            img_dapi_mask = tifffile.imread(path_output_segmentaton + "dapi_maskdapi_" + key_cell_name)

            img_dapi_mask = erase_solitary(img_dapi_mask)

            dye = get_dye(gene_smfish, key_cell_name)
            nb_positive, positive_nuclei = count_positive_cell(dico_stat, key_cell_name, dye)
            average_point_cloud_size = compute_average_size(
                dico_stat[key_cell_name][5]) if dye == 'Cy3' else compute_average_size(dico_stat[key_cell_name][6])
            print(stop)
    inverted_mask = np.ones(img_dapi_mask.shape) - (img_dapi_mask != 0).astype(np.int)
    if len(img_dapi_mask.shape) == 3:
        distance = ndi.distance_transform_edt(inverted_mask, sampling=[voxel_size_z, voxel_size_yx, voxel_size_yx])
    else:
        distance = ndi.distance_transform_edt(inverted_mask)  # compute distance map to border
    labels = watershed(distance, img_dapi_mask)