import streamlit as st; import streamlit.components.v1 as components; from stqdm import stqdm
import os; from os import listdir; import pandas as pd; import numpy as np; from numpy import asarray
import h5py; from pathlib import Path; import pickle; import tensorflow as tf; import copy; import warnings
import PIL; from PIL import Image; PIL.Image.MAX_IMAGE_PIXELS = 933120000
from io import StringIO

import shap; shap.initjs(); import lime; from lime import lime_image; import lime.lime_tabular
import stlearn; from tqdm import tqdm; import math
import plotly.express as px; import matplotlib.pyplot as plt; from matplotlib import cm as cm
import cv2; import joblib; from sklearn.metrics import confusion_matrix; import seaborn as sns

from tensorflow import keras; #from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.applications import VGG16, ResNet50, inception_v3, DenseNet121
from tensorflow.keras.applications import imagenet_utils; from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image as load_img; from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Input, Lambda
from tensorflow.keras.preprocessing import image as image_fun
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as pi;

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, LabelBinarizer; 
from sklearn.metrics import roc_auc_score, plot_confusion_matrix; 
from sklearn.ensemble import RandomForestClassifier; from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import AgglomerativeClustering

import scipy as sp; from scipy import ndimage as ndi; import time
from skimage.feature import peak_local_max; from skimage.segmentation import watershed; from skimage.measure import label
import skimage; from skimage.color import rgb2hed; from skimage.morphology import area_opening; from skimage.segmentation import mark_boundaries

import zipfile
import tempfile
#%%
#D:/Onkar_D/UQ/Project_Spt_Transcriptomics/Output_files/CNN_NB_8genes_model.h5

Path = "D:/Onkar_D/UQ/Project_Spt_Transcriptomics/"

#%%
st.set_page_config(
    page_title="Web-App STimage",

    initial_sidebar_state="auto",

)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

title_container = st.container()
col1, mid, col2 = st.columns([2, 2, 2])

with title_container:
        st.markdown('<h4 style="color: white;">Predicting Spatial Gene Expression Using Tissue Morphology and Spatial Transcriptomics',
                            unsafe_allow_html=True)
        with mid:
            st.markdown('<h5 style="color: lime;">Web-App STimage',
                                unsafe_allow_html=True)

@st.cache
def model_predict_gene(gene):
    i = gene_list.index(gene)
    def combine_model_predict(tile1):
        feature1 = resnet_model.predict(tile1)
        prediction = model_classifier.predict_proba(feature1)#[0]
        return prediction[i]#.reshape(-1,1)
    return combine_model_predict

@st.cache
def model_predict_2gene(gene, tile2):
    i = gene_list.index(gene)

    def combine_model_predict2(tile1):
        feature1 = resnet_model.predict(tile1)
        feature2 = resnet_model.predict(tile2)
        feature = np.concatenate((feature1, feature2), axis=1)
        prediction = model_classifier.predict_proba(feature)
        return prediction[i]
    return combine_model_predict2

@st.cache
def transform_img_fn(img):
    x = np.asarray(img)
    return x

@st.cache
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

#@st.cache
def negative_binomial_layer(x):

    num_dims = len(x.get_shape())
    n, p = tf.unstack(x, num=2, axis=-1)
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)
    n = tf.keras.activations.softplus(n)
    p = tf.keras.activations.sigmoid(p)
    out_tensor = tf.concat((n, p), axis=num_dims - 1)

    return out_tensor

#@st.cache
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
#@st.cache
def CNN_NB_multiple_genes(tile_shape, n_genes):
    tile_input = Input(shape=tile_shape, name="tile_input")
    resnet_base = ResNet50(input_tensor=tile_input, weights='imagenet', include_top=False)

    for i in resnet_base.layers:
        i.trainable = False
    cnn = resnet_base.output
    cnn = GlobalAveragePooling2D()(cnn)

    output_layers = []
    for i in range(n_genes):
        output = Dense(2)(cnn)
        output_layers.append(Lambda(negative_binomial_layer, name="gene_{}".format(i))(output))

    model = Model(inputs=tile_input, outputs=output_layers)
    return model

@st.cache
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

#%%

app_mode = st.sidebar.selectbox("Please select from the following:", ["Tiling, Normalisation and Gene Expression Prediction", "LIME" ,"Meet the Team"])

if app_mode == "Tiling, Normalisation and Gene Expression Prediction":

    st.write("""
                ###### STimage aims at developing a deep learning based computational model to integrate histopathological images with spatial transcriptomics data to predict the spatial gene expression patterns as hallmarks for the detection and classification of cancer cells in a tissue.  
                """)
        
    expander_model_img = st.expander("Model Overview of STimage:")
    expander_model_img.image(Image.open(Path+'Web_App/STimage-web_app_image1.png'), caption='Model Architecture',width=600)


    st.sidebar.write(" ------ ")
    st.sidebar.warning("Enter config file for generating normalised tiles and predicting gene expression!")

    #st.write("""###### """)    
    config_file = st.file_uploader("Config File", type="ini")

    
    expander_norm = st.expander("More about Normalisation of STimage:")
    st.write("""""")
    expander_norm.image(Image.open(Path+'Web_App/Stimage_web_app_data_norm.png'), caption='Data Augmentation and Normalisation',width=600)
    st.write("""""")
    
        

    #%%
if app_mode =="LIME":

    st.sidebar.warning("Upload Image Tile, the file of Trained Model and Enter Gene Names:")

    expander_lime = st.expander('LIME Model:')
    expander_lime.image(Image.open(Path+'Web_App/Lime_Classification.png'), caption='STimage Team',width=600)
    
    #st.sidebar.subheader("Options")

    Top_500genes_train = st.text_input("Space separated Gene Names").split() #['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
    st.subheader("List of Genes")
    st.write(Top_500genes_train)

    st.subheader("Uploading the Image Tiles")
    uploaded_image_file = st.file_uploader("Select image tile:")
    if uploaded_image_file is not None:
        image1 = uploaded_image_file
        Image_tile1 = Image.open(image1)#Image.open
        st.image(Image_tile1)
        image1_trans = transform_img_fn(Image_tile1)
        

    lime_option = st.sidebar.selectbox("Choose the option:", ["Classification", "Regression"])
    if lime_option == "Classification":    

        st.markdown("""
                ## LIME Classification Model
            """)

        resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")

        uploaded_filenames = st.file_uploader('Choose the trained model pickle files:',accept_multiple_files=True)
        for i in uploaded_filenames:
            model_classifier = joblib.load(uploaded_filenames[i])
            model_can_non_can = joblib.load(uploaded_filenames[i+1])
            
            gene_list = Top_500genes_train
            col4, col5 = st.columns(2)
            with col4:
                user_input = gene_list[0]#st.text_input("Enter Gene Name:", 'COX6C')
                
            with col5:
                user_input_2 = gene_list[1]#st.text_input("Enter Gene Name:", 'CD74')
            
            st.success("Gene Specific Nuclei are being computed by LIME")
            st.write(" ")
            st.write("Gene1 and Gene2 are:      ", user_input, "  and   ", user_input_2 )
            
            st.markdown("""
                            #### Watershed Segmentation for LIME
                        """)

            explainer = lime_image.LimeImageExplainer()
            
            
            explanation_gene1 = explainer.explain_instance(image1_trans, model_predict_gene(user_input), top_labels=2, hide_color=0, num_samples=100, segmentation_fn=watershed_segment)
            explanation_gene2 = explainer.explain_instance(image1_trans, model_predict_gene(user_input_2), top_labels=2, hide_color=0, num_samples=100, segmentation_fn=watershed_segment)

        
            col6, col7 = st.columns(2)
        
            with col6:
                
                dict_heatmap1 = dict(explanation_gene1.local_exp[1])
                heatmap1 = np.vectorize(dict_heatmap1.get)(explanation_gene1.segments) 
                fig, ax = plt.subplots()
                plt.imshow(image1_trans)
                plt.imshow(heatmap1, alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap1.max(), vmax = heatmap1.max())
                st.write(fig)
                plt.colorbar()
                plt.show()
            
            with col7:
        
                dict_heatmap2 = dict(explanation_gene2.local_exp[1])
                heatmap2 = np.vectorize(dict_heatmap2.get)(explanation_gene2.segments) 
                fig, ax = plt.subplots()
                plt.imshow(image1_trans)
                plt.imshow(heatmap2, alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap2.max(), vmax = heatmap2.max())
                st.write(fig)
                plt.colorbar()
                plt.show()

#%%
    if lime_option == "Regression":

        st.markdown("""
                        ## LIME Regression Model
                    """)

        class PrinterCallback(tf.keras.callbacks.Callback):
                    
            def on_epoch_end(self, epoch, logs=None):
                print('EPOCH: {}, Train Loss: {}, Val Loss: {}'.format(epoch,
                                                                    logs['loss'],
                                                                    logs['val_loss']))
        
            def on_epoch_begin(self, epoch, logs=None):
                print('-' * 50)
                print('STARTING EPOCH: {}'.format(epoch))
        
        
        
        
        filename = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')#st.text_input('Enter the path of saved model weights:')
        if filename is not None:
            #st.warning("Enter h5-File Path")

            myzipfile = zipfile.ZipFile(filename)
            with tempfile.TemporaryDirectory() as tmp_dir:
                myzipfile.extractall(tmp_dir)
                root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
                model_dir = os.path.join(tmp_dir, root_folder)
                #model = tf.keras.models.load_model(model_dir)

                model_weights = model_dir
                model = CNN_NB_multiple_genes((299, 299, 3), 8)
                model.load_weights(model_weights)
                model.compile(loss=negative_binomial_loss,
                            optimizer=tf.keras.optimizers.Adam(0.0001))

                st.markdown("""
                        #### Watershed Segmentation for LIME
                    """)
                    
                col8, col9 = st.columns(2)         
                    
                explanation_gene1_reg = explainer.explain_instance(image1_trans, model_predict_gene_reg(user_input), 
                                            top_labels=2, num_samples=100,
                                            segmentation_fn=watershed_segment)
                
                explanation_gene2_reg = explainer.explain_instance(image1_trans, model_predict_gene_reg(user_input_2), 
                                            top_labels=2, num_samples=100,
                                            segmentation_fn=watershed_segment)
                    
                with col8:
                    
                    dict_heatmap1 = dict(explanation_gene1_reg.local_exp[explanation_gene1_reg.top_labels[0]])
                    heatmap1 = np.vectorize(dict_heatmap1.get)(explanation_gene1_reg.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(image1_trans)
                    plt.imshow(heatmap1, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap1.max(), vmax = heatmap1.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                
                with col9:
            
                    dict_heatmap2 = dict(explanation_gene2_reg.local_exp[explanation_gene2_reg.top_labels[0]])
                    heatmap2 = np.vectorize(dict_heatmap2.get)(explanation_gene2_reg.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(image1_trans)
                    plt.imshow(heatmap2, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap2.max(), vmax = heatmap2.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()



#%%
if app_mode == "Meet the Team":

    team_img = Image.open(Path+'Web_App/Meet_the_team.png')
    st.image(team_img, caption='Model Architecture',width=600)
    
    expander_email = st.expander("Please contact Dr Quan Nguyen , Xiao Tan or Onkar Mulay!")
    expander_email.write('quan.nguyen@uq.edu.au')
    expander_email.write('xiao.tan@uq.edu.au')
    expander_email.write('o.mulay@uq.net.au')