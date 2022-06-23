
I put a set of zoom images of spots detection in the notebook Workflow.
You can run this notebook to generate other zoom  images. You just have to convert the .czi in tiff, we can use the comand line detail in 1)







### 1) generate tiff data
use the file czi_to_tiff.py with the command 



./czi_to_tiff.py -ptz /home/thomas/Bureau/phd/dropbox/data1/wetransfer-4f06be/ -ptp /home/thomas/Bureau/phd/first_one/

where '-ptz' = "--path_to_czi_folder",  and '-ptp' = "--path_to_project" is where tiff_data are written. a folder tiff data is created




### 2) segment nuclei

./run_seg.py -pi /home/thomas/Bureau/phd/tiff_data/dapi/ -po /home/thomas/Bureau/phd/tiff_data/ -ft 0.8 -m 1



### 3) rna-nuclei association.

use the file spot_detection.py 

./spot_detection.py --path_to_mask_dapi /home/thomas/Bureau/phd/tiff_data/predicted_mask_dapi/   --path_to_af647 /home/thomas/Bureau/phd/tiff_data/af647/
 --path_to_af568 /home/thomas/Bureau/phd/tiff_data/af568/ 
 --seg_3d False


it creates a folder  detected_spot/ which contain a dictionary for each FoV, the dictionary associate each spot (key) to a nuclei (value = nuclei_id, position, distance to the spot).
