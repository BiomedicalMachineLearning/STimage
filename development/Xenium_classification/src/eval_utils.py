import numpy as np
from pathlib import Path
import glob
import os
import shutil

import cv2
import numpy as np
import scipy.io as sio
import argparse

def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)
    return a

# A helper function to unique PanNuke instances indexes to [0..N] range where 0 is background
def map_inst(inst):
    seg_indexes = np.unique(inst)
    new_indexes = np.array(range(0, len(seg_indexes)))
    dict = {}
    for seg_index, new_index in zip(seg_indexes, new_indexes):
        dict[seg_index] = new_index

    return flat_for(inst, lambda x: dict[x])

def create_inst_map(mask_pred):
    if mask_pred.shape[0] < mask_pred.shape[1]:
        mask_pred = np.transpose(mask_pred, (1,2,0))
    inst = np.zeros((256,256))
    for j in range(mask_pred.shape[2] - 1):
        #copy value from new array if value is not equal 0
        inst = np.where(mask_pred[:,:,j] != 0, mask_pred[:,:,j], inst)
    map_inst(inst)
    return inst

def create_type_map(mask_pred):
    # Input is (C,X,Y)
    if len(mask_pred.shape) == 4:
        raise ValueError('Does not support batched instances')
    # swap background to first, so that the argmax returns N for the Nth class
    msks = np.concatenate((mask_pred[[-1]], mask_pred[:-1]), axis=0)
    return np.argmax(msks, axis=0)

# The following code should do the same thing, but must have mask_pred as (X,Y,C)
def create_type_map2(mask_pred):
    if len(mask_pred.shape) == 4:
        raise ValueError('Does not support batched instances')
    # Input is (X,Y,C)
    if mask_pred.shape[0] < mask_pred.shape[1]:
        mask_pred = np.moveaxis(mask_pred, 0, 2)
    types = np.zeros((256,256))
    for j in range(mask_pred.shape[2] - 1):
        # write type index if mask is not equal 0 and value is still 0
        types = np.where((mask_pred[:,:,j] != 0) & (types == 0), j+1, types)
    return types

def save_mat(path_to_folder, mask_pred):
    #path_to_folder = "/home/ccurs011/HoverNet/PanTransform/fold3"
    if isinstance(path_to_folder, str):
        path_to_folder = Path(path_to_folder)
    # path_to_folder = "88_pan/centroisd/"
    
    # read all .npy files from directory
    # files = []
    # for i in os.listdir(path_to_folder):
    #     if i.endswith('.npy'):
    #         files.append(np.load(path_to_folder + "/" + i))
    
    def get_inst_centroid(inst_map):
        inst_centroid_list = []
        inst_id_list = list(np.unique(inst_map))
        for inst_id in inst_id_list[1:]: # avoid 0 i.e background
            mask = np.array(inst_map == inst_id, np.uint8)
            inst_moment = cv2.moments(mask)
            inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                             (inst_moment["m01"] / inst_moment["m00"])]
            inst_centroid_list.append(inst_centroid)
        return np.array(inst_centroid_list)
    
    result_dict_list= list()
    
    # for j, file in enumerate(files):
    #     inst_map = file[:,:,3]
    #     type_map = file[:,:,4]
    for j, file in enumerate(mask_pred):
        # print(j)
        inst_map = create_inst_map(file)
        type_map = create_type_map(file)
        
        inst_centroid = get_inst_centroid(inst_map)
        inst_type = np.zeros((inst_centroid.shape[0], 1))
        for i in range(inst_centroid.shape[0]):
            x = int(round(inst_centroid[i,0]))
            y = int(round(inst_centroid[i,1]))
            inst_type[i] = type_map[y, x]
            
        result_dict = dict()
        result_dict['inst_centroid'] = inst_centroid
        result_dict['inst_type'] = inst_type
        result_dict['inst_map'] = inst_map
        result_dict_list.append(result_dict)
        sio.savemat(path_to_folder / f"{str(j+1)}.mat", result_dict)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#             "--out-dir", type=str, default=None, help="Path to output directory containing masks.npy, truths.npy, types.npy"
#     )
#     parser.add_argument(
#             "--save_path", type=str, default=None, help="Path to save output"
#     )
#     parser.add_argument(
#             "--class_path", type=str, default=None, help="Path to anndata object containing the classes"
#     )
#     args = parser.parse_args()
#     print(args)
#     main(args)

# save_mat(out_dir + "centroids_truth/" , mask_truth)
# save_mat(out_dir + "centroids_pred/" , mask_pred)