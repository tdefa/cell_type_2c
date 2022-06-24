

# All the usefull function and the pipeline is in main_cluster_split

# to run it:

##### activate your conda env ex: 

optional arguments:
  -h, --help            show this help message and exit
  -ptz PATH_TO_CZI_FOLDER, --path_to_czi_folder PATH_TO_CZI_FOLDER
                        path to the folder containing the czi
                        
  --list_folder LIST_FOLDER [LIST_FOLDER ...]
                        list of folders in the czi folders to analyse
                        
  --new_probe NEW_PROBE [NEW_PROBE ...]
                        command to add new probes or change parameters of existing one to add it do --new_probe p1 epsi overlapping
                        --new_probe p2 20 0.3 where 'epsi' is the parameter of the dbscan 'overlapping' is the percent of overlap to make
                        a cell positive to a probe
                        
  --manual_threshold_cy3 MANUAL_THRESHOLD_CY3
                        write a json like the : {"02_NI1230_Lamp3-Cy5_Pdgfra-Cy3_08.tiff": 8, "01_IR5M1236_Lamp3-Cy5_Pdgfra-Cy5_04.tiff":
                        7} to set manually the rna spot detection threshold
                        
                        
  --manual_threshold_cy5 MANUAL_THRESHOLD_CY5
  
  
  
  -dns DICO_NAME_SAVE, --dico_name_save DICO_NAME_SAVE
                        additional name in the save result

  
  
                        do : prepare_czi to tiff
  -sg SEGMENTATION, --segmentation SEGMENTATION
                        do segmentation
  --spot_detection SPOT_DETECTION
                        do spots detection
  --classify CLASSIFY   do classification / cell type mapping
  --save_plot SAVE_PLOT do save plot




###################

helping function 
