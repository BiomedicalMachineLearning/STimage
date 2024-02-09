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
from PIL import Image
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


# LIME Functions
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


def model_predict_gene_reg(gene):
    i = gene_list.index(gene)
    from scipy.stats import nbinom
    def model_predict(x):
        test_predictions = model.predict(x)
        n = test_predictions[i][:, 0]
        p = test_predictions[i][:, 1]
        y_pred = nbinom.mean(n, p)
        return y_pred.reshape(-1,1)
    return model_predict


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
        image = np.asarray(image_fun.load_img(i))
        try:
            explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=2, hide_color=0, num_samples=500, segmentation_fn=watershed_segment)
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


def LIME_reg_exceptional_handling(image_path, gene):
    LIME_heatmaps = []; no_segments = []
    for i in image_path:
        image = np.asarray(image_fun.load_img(i))
        try:
            explanation = explainer.explain_instance(
                image,
                model_predict_gene_reg(gene),
                top_labels=1,
                hide_color=0,
                num_samples=250,
                segmentation_fn=watershed_segment)
            dict_heatmap = dict(explanation.local_exp[explanation.top_labels[0]])
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


def CNN_NB_multiple_genes(tile_shape, n_genes, ft=False):
    tile_input = Input(shape=tile_shape, name="tile_input")
    cnn_base = ResNet50(input_tensor=tile_input, weights='imagenet', include_top=False)
    
    if not ft:
        for i in cnn_base.layers:
            i.trainable = False
    cnn = cnn_base.output
    cnn = GlobalAveragePooling2D()(cnn)

    output_layers = []
    for i in range(n_genes):
        output = Dense(2)(cnn)
        output_layers.append(Lambda(negative_binomial_layer, name="gene_{}".format(i))(output))

    model = Model(inputs=tile_input, outputs=output_layers)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    model.compile(loss=negative_binomial_loss,
                  optimizer=optimizer)
    return model

def negative_binomial_layer(x):
    num_dims = len(x.get_shape())
    n, p = tf.unstack(x, num=2, axis=-1)
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)
    n = tf.keras.activations.softplus(n)
    p = tf.keras.activations.sigmoid(p)
    out_tensor = tf.concat((n, p), axis=num_dims - 1)
    return out_tensor

def negative_binomial_loss(y_true, y_pred):
    n, p = tf.unstack(y_pred, num=2, axis=-1)
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)
    nll = (
            tf.math.lgamma(n)
            + tf.math.lgamma(y_true + 1)
            - tf.math.lgamma(n + y_true)
            - n * tf.math.log(p)
            - y_true * tf.math.log(1 - p)
    )
    return nll


Path = "/scratch/project/stseq/Onkar/STimage_v1/"
explainer = lime_image.LimeImageExplainer()

# Load Tiles
data_FFPE = sc.read_h5ad("/QRISdata/Q1851/Xiao/Wiener_backup/STimage_exp/stimage_LOOCV_9visium_selected_gene/pred_FFPE.h5ad")
data_FFPE.obs["tile_path"] = Path+"tiles/tiles/"+data_FFPE.obs["tile_path"].str.split('/', expand=True)[7]
Selection_region_FFPE = list(data_FFPE.obs["tile_path"])
gene_list = list(data_FFPE.var_names)

# Load model
model_weights = "/QRISdata/Q1851/Xiao/Wiener_backup/STimage_exp/stimage_LOOCV_9visium_selected_gene/stimage_model_FFPE.h5"
model = CNN_NB_multiple_genes((299, 299, 3), 1522)
model.load_weights(model_weights)
model.compile(loss=negative_binomial_loss,  
            optimizer=tf.keras.optimizers.Adam(0.0001))

# Run LIME
LIMEMask_FFPE_CD24 = LIME_reg_exceptional_handling(Selection_region_FFPE,gene_list[551])
np.save(Path+"pickle/LIMEMaskReg_FFPE_CD24_all.npy",LIMEMask_FFPE_CD24)

LIMEMask_FFPE_CD52 = LIME_reg_exceptional_handling(Selection_region_FFPE,gene_list[26])
np.save(Path+"pickle/LIMEMaskReg_FFPE_CD52_all.npy",LIMEMask_FFPE_CD52)
