import shap
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image as image_fun
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda
import numpy as np

import skimage
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
import scipy as sp
from scipy import ndimage as ndi
from skimage.morphology import area_opening
import math
from sklearn.linear_model import LinearRegression, SGDRegressor
from lime import lime_image
from PIL import Image
from matplotlib import pyplot as plt
import pickle

def CNN_NB_multiple_genes(tile_shape, n_genes, cnnbase="resnet50", ft=False):

    tile_input = Input(shape=tile_shape, name="tile_input")
    if cnnbase == "resnet50":
        cnn_base = ResNet50(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "vgg16":
        cnn_base = VGG16(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "inceptionv3":
        cnn_base = InceptionV3(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "mobilenetv2":
        cnn_base = MobileNetV2(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "densenet121":
        cnn_base = DenseNet121(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "xception":
        cnn_base = Xception(input_tensor=tile_input, weights='imagenet', include_top=False)
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
def negative_binomial_loss(y_true, y_pred):

    # Separate the parameters
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
def negative_binomial_layer(x):

    num_dims = len(x.get_shape())
    n, p = tf.unstack(x, num=2, axis=-1)
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)
    n = tf.keras.activations.softplus(n)
    p = tf.keras.activations.sigmoid(p)
    out_tensor = tf.concat((n, p), axis=num_dims - 1)
    return out_tensor
def model_predict_gene(gene):
    i = gene_list.index(gene)
    from scipy.stats import nbinom
    def model_predict(x):
        test_predictions = model.predict(x)
        n = test_predictions[i][:, 0]
        p = test_predictions[i][:, 1]
        y_pred = nbinom.mean(n, p)
        return y_pred.reshape(-1,1)
    return model_predict
def watershed_segment(image):
    annotation_hed = rgb2hed(image)
    annotation_h = annotation_hed[:,:,0]
    annotation_h *= 255.0 / np.percentile(annotation_h, q=0.01)
    thresh = skimage.filters.threshold_otsu(annotation_h)*0.7
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        annotation_h < thresh
    )
    distance = ndi.distance_transform_edt(im_fgnd_mask)
    coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=im_fgnd_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(annotation_h, markers, mask=im_fgnd_mask)
    im_nuclei_seg_mask = area_opening(labels, area_threshold=64).astype(np.int)
    map_dic = dict(zip(np.unique(im_nuclei_seg_mask), np.arange(len(np.unique(im_nuclei_seg_mask)))))
    im_nuclei_seg_mask = np.vectorize(map_dic.get)(im_nuclei_seg_mask)
    return im_nuclei_seg_mask
def LIME_reg(image_path, gene):
    LIME_masks = []; LIME_heatmaps = []
    for i in image_path:
        image = np.asarray(image_fun.load_img(i))
        explanation = explainer.explain_instance(image, model_predict_gene_reg(gene), top_labels=1, hide_color=0, num_samples=100, segmentation_fn=watershed_segment)
        temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
        mask[mask<=0] = 0
        LIME_masks.append(mask)
        dict_heatmap = dict(explanation.local_exp[0])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        LIME_heatmaps.append(heatmap)
    return LIME_masks, LIME_heatmaps
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
def model_predict_gene_nb_kernel(gene):
    i = gene_list.index(gene)
    from scipy.stats import nbinom
    def model_predict(z):
        test_predictions = model.predict(mask_image(z, segments_slic, image_orig, 0))
        n = test_predictions[i][:, 0]
        p = test_predictions[i][:, 1]
        y_pred = nbinom.mean(n, p)
        return y_pred.reshape(-1,1)
    return model_predict
def LIME(image_path, gene):
    LIME_heatmaps = []
    for i in image_path:
        image = np.asarray(image_fun.load_img(i))
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image, model_predict_gene(gene), top_labels=1, hide_color=0, num_samples=1000, segmentation_fn=watershed_segment)
        dict_heatmap = dict(explanation.local_exp[0])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        LIME_heatmaps.append(heatmap)
    return LIME_heatmaps
def scale_mask(mat,N):
    mat = mat.reshape(-1,N*89401)
    mat_std = (mat - mat.min()) / (mat.max() - mat.min())
    mat_scaled = mat_std * (1 - 0) + 0
    return mat_scaled.reshape(N,299,299)



gene_list = ["IGHG3", "IGHM", "C3", "AP2B1", "GNAS", "PRLR", "PUM1"]
n_genes = len(gene_list)
model = CNN_NB_multiple_genes((299, 299, 3), n_genes)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                            restore_best_weights=False)
model.load_weights("/afm01/UQ/Q2051/STimage_project/pretrained_model/CNN_NB_cancer_immune_7genes.h5")


Path = "/home/uqomulay/90days/STimage_outputs/SHAP_LIMEREG"
image_dir = np.load('/home/uqomulay/90days/STimage_outputs/SHAP_LIME/image_dir',allow_pickle=True)
explainer = lime_image.LimeImageExplainer()


FFPE_C3_LIMEREGMask = LIME(image_dir,gene_list[2])
FFPE_C3_LIMEREGMask_scaled = scale_mask(np.array(FFPE_C3_LIMEREGMask),25)
np.save(Path+"/FFPE_C3_LIMEREGMask_scaled.npy",FFPE_C3_LIMEREGMask_scaled)

FFPE_GNAS_LIMEREGMask = LIME(image_dir,gene_list[4])
FFPE_GNAS_LIMEREGMask_scaled = scale_mask(np.array(FFPE_GNAS_LIMEREGMask),25)
np.save(Path+"/FFPE_GNAS_LIMEREGMask_scaled.npy",FFPE_GNAS_LIMEREGMask_scaled)
    
shap_values_GNAS_agnostic = []; shap_segments_GNAS_agnostic = []
for i in range(0,len(image_dir)):
    image = Image.open(image_dir[i])
    image_orig = img_to_array(image)
    segments_slic = watershed_segment(image)
    shap_segments_GNAS_agnostic.append(segments_slic)
    explainer = shap.KernelExplainer(model_predict_gene_nb_kernel("GNAS"), np.zeros((1,shap_segments_GNAS_agnostic[i].max())))
    shap_values_GNAS_agnostic.append(explainer.shap_values(np.ones((1,shap_segments_GNAS_agnostic[i].max())), nsamples=shap_segments_GNAS_agnostic[i].max()))
    
with open(Path+'/shap_segments_GNAS_agnostic', "wb") as fp:
    pickle.dump(shap_segments_GNAS_agnostic, fp)
with open(Path+'/shap_values_GNAS_agnostic', "wb") as fp:
    pickle.dump(shap_values_GNAS_agnostic, fp)
    
shap_values_C3_agnostic = []; shap_segments_C3_agnostic = []
for i in range(0,len(image_dir)):
    image = Image.open(image_dir[i])
    image_orig = img_to_array(image)
    segments_slic = watershed_segment(image)
    shap_segments_C3_agnostic.append(segments_slic)
    explainer = shap.KernelExplainer(model_predict_gene_nb_kernel("C3"), np.zeros((1,shap_segments_C3_agnostic[i].max())))
    shap_values_C3_agnostic.append(explainer.shap_values(np.ones((1,shap_segments_C3_agnostic[i].max())), nsamples=shap_segments_C3_agnostic[i].max()))
    
with open(Path+'/shap_segments_C3_agnostic', "wb") as fp:
    pickle.dump(shap_segments_C3_agnostic, fp)
with open(Path+'/shap_values_C3_agnostic', "wb") as fp:
    pickle.dump(shap_values_C3_agnostic, fp)

