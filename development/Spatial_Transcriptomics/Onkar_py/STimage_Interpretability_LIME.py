import json
import shap
import tensorflow as tf
import anndata
from matplotlib import pyplot as plt
import numpy as np
import joblib
from lime import lime_image
import cv2
import pandas as pd
import PIL
from PIL import Image
import random
import scipy as sp
from scipy import ndimage as ndi
import joblib
import sys
import skimage
import pickle

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


Path = "/home/uqomulay/90days/STimage_outputs/"
test_adata = anndata.read_h5ad(Path+"pickle/test_anndata_norm.h5ad")
gene_list = ['CD74', 'CD24', 'CD63', 'CD81', 'CD151', 'C3',
             'COX6C', 'TP53', 'PABPC1', 'GNAS', 'B2M', 'SPARC', 'HSP90AB1', 'TFF3', 'ATP1A1', 'FASN']
test_adata = test_adata[:,gene_list]

F1160920 = test_adata[test_adata.obs["library_id"]=="1160920F"]
cord1 = F1160920.obs[(F1160920.obs['imagecol'] >= 13000) & (F1160920.obs['imagecol'] < 22000)]
cord2 = cord1[(cord1['imagerow'] >= 13000) & (cord1['imagerow'] < 19000)]
F1160920.obs["active"] = np.where(F1160920.obs.index.isin(cord2.index),2,0)
Selection_region_F1160920_1 = random.sample(list(F1160920.obs[F1160920.obs["active"]==2]["tile_path"]),250)

del cord1, cord2
cord1 = F1160920.obs[(F1160920.obs['imagecol'] >= 13000) & (F1160920.obs['imagecol'] < 17500)]
cord2 = cord1[(cord1['imagerow'] >= 7500) & (cord1['imagerow'] < 11000)]
F1160920.obs["active"] = np.where(F1160920.obs.index.isin(cord2.index),2,0)
Selection_region_F1160920_2 = random.sample(list(F1160920.obs[F1160920.obs["active"]==2]["tile_path"]),75)

del cord1, cord2
ffpe = test_adata[test_adata.obs["library_id"]=="FFPE"]
cord1 = ffpe.obs[(ffpe.obs['imagecol'] >= 7000) & (ffpe.obs['imagecol'] < 11000)]
cord2 = cord1[(cord1['imagerow'] >= 10500) & (cord1['imagerow'] < 18000)]
ffpe.obs["active"] = np.where(ffpe.obs.index.isin(cord2.index),2,0)
Selection_region_ffpe_1 = list(ffpe.obs[ffpe.obs["active"]==2]["tile_path"])

del cord1, cord2
cord1 = ffpe.obs[(ffpe.obs['imagecol'] >= 12500) & (ffpe.obs['imagecol'] < 17500)]
cord2 = cord1[(cord1['imagerow'] >= 6000) & (cord1['imagerow'] < 9000)]
ffpe.obs["active"] = np.where(ffpe.obs.index.isin(cord2.index),2,0)
Selection_region_ffpe_2 = list(ffpe.obs[ffpe.obs["active"]==2]["tile_path"])


def watershed_segment(image):
    annotation_hed = rgb2hed(image)
    annotation_h = annotation_hed[:,:,0]
    annotation_h *= 255.0 / np.percentile(annotation_h, q=0.01)
    thresh = skimage.filters.threshold_otsu(annotation_h)*0.7
    im_fgnd_mask = sp.ndimage.binary_fill_holes(
        annotation_h < thresh
    )
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
        explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=2, hide_color=0, num_samples=1000, segmentation_fn=watershed_segment)
        temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
        dict_heatmap = dict(explanation.local_exp[1])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        LIME_heatmaps.append(heatmap)
    return LIME_heatmaps

def LIME_exceptional_handling(image_path, gene):
    LIME_heatmaps = []; no_segments = []
    for i in image_path:
        image = np.asarray(image_fun.load_img(i))
        try:
            explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=2, hide_color=0, num_samples=1000, segmentation_fn=watershed_segment)
            temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
            dict_heatmap = dict(explanation.local_exp[1])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        except:
            no_segments.append(i)
        LIME_heatmaps.append(heatmap)
    print(no_segments)
    return LIME_heatmaps


def SHAP_global(image_path, gene):
    SHAP_mask = []
    for i in image_path:
        masker = shap.maskers.Image("inpaint_telea", (299,299,3))
        explainer_shap = shap.Explainer(model_predict_gene(gene), masker, output_names=STimage_classification_classes)
        shap_values = explainer_shap(np.array(plt.imread(i).astype('float32')).reshape(1,299,299,3), max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:2])
        SHAP_mask.append(shap_values)
    return SHAP_mask


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


def SHAP_scale(ShapMask,N):
    ShapMask_high = []; ShapMask_high_1c = []
    for i in range(0,len(ShapMask)):
        ShapMask_high.append(ShapMask[i].values[:,:,:,:,np.argmax(ShapMask[i].base_values)])
    ShapMask_high = np.array(ShapMask_high).reshape(N,299,299,3)
    for i in range(0,len(ShapMask)):
        ShapMask_high_1c.append(ShapMask_high[i,:,:,:1])
    ShapMask_high_1c = np.array(ShapMask_high_1c)
    return scale_mask(ShapMask_high_1c,N)


explainer = lime_image.LimeImageExplainer()
clf_resnet = joblib.load(Path+'pickle/STimage_LR.pkl')
resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")


FFPE_C3_LIMEMask = LIME(Selection_region_ffpe_1+Selection_region_ffpe_2,gene_list[5])
#FFPE_C3_LIMEMask_scaled = scale_mask(np.array(FFPE_C3_LIMEMask),555)
np.save(Path+"LIME_Agnostic/FFPE_C3_LIMEMask_scaled.npy",FFPE_C3_LIMEMask)


FFPE_GNAS_LIMEMask = LIME(Selection_region_ffpe_1+Selection_region_ffpe_2,gene_list[9])
#FFPE_GNAS_LIMEMask_scaled = scale_mask(np.array(FFPE_GNAS_LIMEMask),555)
np.save(Path+"LIME_Agnostic/FFPE_GNAS_LIMEMask_scaled.npy",FFPE_GNAS_LIMEMask)


F1160920_C3_LIMEMask = LIME_exceptional_handling(Selection_region_F1160920_1+Selection_region_F1160920_2,gene_list[5])
#F1160920_C3_LIMEMask_scaled = scale_mask(np.array(F1160920_C3_LIMEMask),325)
np.save(Path+"LIME_Agnostic/F1160920_C3_LIMEMask_scaled.npy",F1160920_C3_LIMEMask)


F1160920_GNAS_LIMEMask = LIME_exceptional_handling(Selection_region_F1160920_1+Selection_region_F1160920_2,gene_list[9])
#F1160920_GNAS_LIMEMask_scaled = scale_mask(np.array(F1160920_GNAS_LIMEMask),325)
np.save(Path+"LIME_Agnostic/F1160920_GNAS_LIMEMask_scaled.npy",F1160920_GNAS_LIMEMask)

