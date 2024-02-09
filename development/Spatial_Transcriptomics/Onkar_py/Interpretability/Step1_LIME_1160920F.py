import json
import shap
import anndata
from matplotlib import pyplot as plt
import numpy as np
import joblib
from lime import lime_image
import cv2
import pandas as pd
import PIL
from PIL import Image, ImageFilter
import random
import scipy as sp
from scipy import ndimage as ndi
import joblib
import sys
import scanpy as sc
import skimage
import pickle
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from skimage.transform import resize
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries, watershed
from skimage.segmentation import slic
from skimage.morphology import area_opening

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as image_fun
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image as image_fun
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda
import tensorflow as tf


# Load Data
Path="/scratch/project/stseq/Onkar/STimage_v1/Outputs/"
test_adata = sc.read_h5ad(Path+"pickle/h5ad/test_adata_top100_wts.h5ad")
with open(Path+'pickle/pkl/top_100gene.pkl', 'rb') as f:
    gene_list = list(pickle.load(f))
explainer = lime_image.LimeImageExplainer()
clf_resnet = joblib.load(Path+'pickle/pkl/LRmodel_100genes.pkl')
resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")


# LIME Functions
def watershed_segment(image):
    annotation_hed = rgb2hed(image)
    annotation_h = annotation_hed[:,:,0]
    annotation_h *= 255.0 / np.percentile(annotation_h, q=0.01)
    thresh = skimage.filters.threshold_otsu(annotation_h)*1.0
    im_fgnd_mask = sp.ndimage.binary_fill_holes(
        annotation_h < thresh)
    distance = ndi.distance_transform_edt(im_fgnd_mask)
    coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=im_fgnd_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(annotation_h, markers, mask=im_fgnd_mask)
    im_nuclei_seg_mask = area_opening(labels, area_threshold=64).astype(np.int32)
    map_dic = dict(zip(np.unique(im_nuclei_seg_mask), np.arange(len(np.unique(im_nuclei_seg_mask)))))
    im_nuclei_seg_mask = np.vectorize(map_dic.get)(im_nuclei_seg_mask)
    return im_nuclei_seg_mask


def model_predict_gene(gene):
    i = gene_list.index(gene)
    def combine_model_predict(tile1):
        feature1 = resnet_model.predict(tile1)
        prediction = clf_resnet.predict_proba(feature1)
        return prediction[i]
    return combine_model_predict


def LIME(image_path, gene):
    LIME_heatmaps = []
    for i in image_path:
        image = np.asarray(image_fun.load_img(i))
        explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=2, hide_color=0, num_samples=500, segmentation_fn=watershed_segment)
        temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
        dict_heatmap = dict(explanation.local_exp[1])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        LIME_heatmaps.append(heatmap)
    return LIME_heatmaps


def LIME_exceptional_handling(image_path, gene):
    LIME_heatmaps = []; no_segments = []
    for i in image_path:
        image = Image.open(i)
        image = image.filter(ImageFilter.UnsharpMask(radius=2.5, percent=150, threshold=3))
        image = np.asarray(image)
        try:
            explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=2, hide_color=0, num_samples=100, segmentation_fn=watershed_segment)
            temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
            dict_heatmap = dict(explanation.local_exp[1])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        except:
            no_segments.append(i)
            print(f"Error generating LIME explanation for image: {i}")  
            heatmap = np.full((299, 299), 2)
        LIME_heatmaps.append(heatmap)
    print(f"Images with segmentation issues: {no_segments}")
    return LIME_heatmaps


def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out


def model_predict_gene_kernel(gene):
    i = gene_list.index(gene)
    def combine_model_predict(z):
        feature1 = resnet_model.predict(mask_image(z, segments_slic, image_orig, 0))
        prediction = clf_resnet.predict_proba(feature1)
        return prediction[i]
    return combine_model_predict


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

def scale_mask(mat,N):
    mat = mat.reshape(-1,N*89401)
    mat_std = (mat - mat.min()) / (mat.max() - mat.min())
    mat_scaled = mat_std * (1 - 0) + 0
    return mat_scaled.reshape(N,299,299)

# For specific set of tiles
##########################################################################################################################################
# data_1160920F = test_adata[test_adata.obs["library_id"].isin(["1160920F"])]
# cord1 = data_1160920F.obs[(data_1160920F.obs['imagecol'] < 16750) & (data_1160920F.obs['imagecol'] > 14750)]
# cord2 = cord1[(cord1['imagerow'] > 7000) & (cord1['imagerow'] < 8000)]
# data_1160920F.obs["active"] = np.where(data_1160920F.obs.index.isin(cord2.index),1,0)
# Selection_region_1160920F_1 = list(data_1160920F.obs[data_1160920F.obs["active"]==1]["tile_path"])

# del cord1, cord2
# ##########################################################################################################################################
# cord1 = data_1160920F.obs[(data_1160920F.obs['imagecol'] < 15500) & (data_1160920F.obs['imagecol'] > 14500)]
# cord2 = cord1[(cord1['imagerow'] > 18000) & (cord1['imagerow'] < 19000)]
# data_1160920F.obs["active"] = np.where(data_1160920F.obs.index.isin(cord2.index),1,0)
# Selection_region_1160920F_2 = list(data_1160920F.obs[data_1160920F.obs["active"]==1]["tile_path"])

# Load Tiles
Selection_region_1160920F = list(data_1160920F.obs["tile_path"])

# Run LIME Function
LIMEMask_1160920F_CD24 = LIME_exceptional_handling(Selection_region_1160920F,gene_list[8])
np.save(Path+"pickle/npy/LIMEMask_1160920F_CD24_all.npy",LIMEMask_1160920F_CD24)

LIMEMask_1160920F_CD52 = LIME_exceptional_handling(Selection_region_1160920F,gene_list[85])
np.save(Path+"pickle/npy/LIMEMask_1160920F_CD52_all.npy",LIMEMask_1160920F_CD52)