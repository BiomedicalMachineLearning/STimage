import streamlit as st; import streamlit.components.v1 as components; from stqdm import stqdm
import os; from os import listdir; import pandas as pd; import numpy as np; from numpy import asarray
import h5py; from pathlib import Path; import pickle; import tensorflow as tf; import copy; import warnings
import PIL; from PIL import Image; PIL.Image.MAX_IMAGE_PIXELS = 933120000
from io import StringIO

import shap; shap.initjs(); import lime; from lime import lime_image; from lime import lime_tabular

from tqdm import tqdm; import math
import matplotlib.pyplot as plt; from matplotlib import cm as cm
import cv2
import joblib
import seaborn as sns

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.applications import VGG16, ResNet50, inception_v3, DenseNet121, imagenet_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Input, Lambda
from tensorflow.keras.preprocessing import image as image_fun


from sklearn import preprocessing; from sklearn.preprocessing import LabelEncoder, LabelBinarizer; 
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier; from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

import scipy as sp; from scipy import ndimage as ndi
import time
import skimage; from skimage.feature import peak_local_max; from skimage.segmentation import watershed; from skimage.measure import label
from skimage.color import rgb2hed; from skimage.morphology import area_opening; from skimage.segmentation import mark_boundaries
from scipy.stats import nbinom

import zipfile
import tempfile

from interpretability_functions import *

#####
#Preprocessing
import argparse
import configparser
from pathlib import Path
import sys
#from _img_normaliser import IterativeNormaliser
#from _model import ResNet50_features
#from _utils import tiling, ensembl_to_id, ReadOldST, Read10X, scale_img, calculate_bg, classification_preprocessing
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as pi;
#from tensorflow.keras.utils import np_utils
#####

#%%

Path = "/utils/images/"

with open('/utils/pkl/top_100gene.pkl', 'rb') as f:
    gene_list_clf = list(pickle.load(f))
    
with open('/utils/pkl/top_300gene_reg.pkl', 'rb') as f:
    gene_list_reg = list(pickle.load(f))

#%%
######################################################################################################################################################################
# Setting up the page
######################################################################################################################################################################

st.set_page_config(page_title="Web-App STimage", initial_sidebar_state="auto",)

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

title_container_1 = st.container()
col1, mid, col2 = st.columns([2, 2, 2])
with title_container_1:
        with mid:
            st.markdown('<h3 style="color: purple;"> STimage Web-App',
                                unsafe_allow_html=True)
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            
title_container_2 = st.container()
with title_container_2:
        st.markdown('<h5 style="color: black;"> STimage Model Interpretability Tool: Visualise SHAP and LIME scores for genes of interest ',
                                unsafe_allow_html=True)
        st.write('')
        
######################################################################################################################################################################
# Setting up the functions
######################################################################################################################################################################

@st.cache
def model_predict_gene_ml(gene):
    i = gene_list_clf.index(gene)
    def combine_model_predict(tile1):
        feature1 = resnet_model.predict(tile1)
        prediction = model_c.predict_proba(feature1)
        gene_prediction = prediction[i]
        return gene_prediction
    return combine_model_predict

@st.cache #(suppress_st_warning=True)
def LIME(image_path, gene):
    image = np.asarray(Image.open(image_path))
    explanation = explainer.explain_instance(
        image,
        model_predict_gene_ml(gene),
        top_labels=2,
        hide_color=0,
        num_samples=100,
        segmentation_fn=watershed_segment)
    temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
    dict_heatmap = dict(explanation.local_exp[1])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    return heatmap

@st.cache
def model_predict_gene_reg(gene):
    i = gene_list_reg.index(gene)
    from scipy.stats import nbinom
    def model_predict(x):
        test_predictions = model.predict(x)
        n = test_predictions[i][:, 0]
        p = test_predictions[i][:, 1]
        y_pred = nbinom.mean(n, p)
        return y_pred.reshape(-1,1)
    return model_predict

@st.cache #(suppress_st_warning=True)
def LIME_reg(image_path, gene):
    image = np.asarray(Image.open(image_path))
    explanation = explainer.explain_instance(
        image,
        model_predict_gene_reg(gene),
        top_labels=1,
        hide_color=0,
        num_samples=100,
        segmentation_fn=watershed_segment)
    # temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
    # dict_heatmap = dict(explanation.local_exp[1])
    dict_heatmap = dict(explanation.local_exp[explanation.top_labels[0]])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    return heatmap

def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))  # Calculate mean across spatial dimensions
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    return out

@st.cache
def model_predict_gene_kernel(gene):
    i = gene_list.index(gene)  # Find the index of the gene in the list
    def combine_model_predict(z):
        masked_image = mask_image(z, segments_slic, image_orig, 0)
        feature1 = resnet_model.predict(masked_image)
        prediction = model_c.predict_proba(feature1)
        gene_probabilities = prediction[i]
        return gene_probabilities
    return combine_model_predict

@st.cache
def SHAP_Classification(segments_slic,image_orig,gene_input):
    for i in range(0,1):
        ffpe_shap_values_agnostic = []; ffpe_shap_segments_agnostic = []        
        ffpe_shap_segments_agnostic.append(segments_slic)
        explainer = shap.KernelExplainer(model_predict_gene_kernel(gene_input), np.zeros((1,ffpe_shap_segments_agnostic[i].max())))
        ffpe_shap_values_agnostic.append(explainer.shap_values(np.ones((1,ffpe_shap_segments_agnostic[i].max())), nsamples=ffpe_shap_segments_agnostic[i].max()))

    ffpe_shap_segments_agnostic_scores = []
    for j in range(0,1):
        out = np.zeros((299,299))
        for i in range(0,ffpe_shap_segments_agnostic[j].max()):
            out[ffpe_shap_segments_agnostic[j] == i] = ffpe_shap_values_agnostic[j][0][0][i]
        ffpe_shap_segments_agnostic_scores.append(out)
    return ffpe_shap_segments_agnostic_scores

@st.cache
def model_predict_gene_kernel_reg(gene):
    i = gene_list.index(gene)  # Find the index of the gene in the list
    def combine_model_predict(z):
        masked_image = mask_image(z, segments_slic, image_orig, 0)
        test_predictions = model.predict(masked_image)
        n = test_predictions[i][:, 0]
        p = test_predictions[i][:, 1]
        y_pred = nbinom.mean(n, p)
        return y_pred.reshape(-1,1)
    return combine_model_predict

@st.cache
def SHAP_reg(segments_slic,image_orig,gene_input):
    for i in range(0,1):
        ffpe_shap_values_agnostic = []; ffpe_shap_segments_agnostic = []        
        ffpe_shap_segments_agnostic.append(segments_slic)
        explainer = shap.KernelExplainer(model_predict_gene_kernel_reg(gene_input), np.zeros((1,ffpe_shap_segments_agnostic[i].max())))
        ffpe_shap_values_agnostic.append(explainer.shap_values(np.ones((1,ffpe_shap_segments_agnostic[i].max())), nsamples=ffpe_shap_segments_agnostic[i].max()))

    ffpe_shap_segments_agnostic_scores = []
    for j in range(0,1):
        out = np.zeros((299,299))
        for i in range(0,ffpe_shap_segments_agnostic[j].max()):
            out[ffpe_shap_segments_agnostic[j] == i] = ffpe_shap_values_agnostic[j][0][0][i]
        ffpe_shap_segments_agnostic_scores.append(out)
    return ffpe_shap_segments_agnostic_scores

@st.cache
def make_global():
    global image1
    global gene_input_1
    global gene_input_2
    global gene_list
    global model_c
    global model
    global resnet_model

# session_state = st.session_state.setdefault('session_state', init_session_state())
if "image_var" not in st.session_state:
    st.session_state.image_var = None
if "gene_1" not in st.session_state:
    st.session_state.gene_1 = None
if "gene_2" not in st.session_state:
    st.session_state.gene_2 = None
if "gene_list" not in st.session_state:
    st.session_state.gene_list = None
if "model_clf" not in st.session_state:
    st.session_state.model_clf = None
if "model_reg" not in st.session_state:
    st.session_state.model_reg = None
if "resnet50" not in st.session_state:
    st.session_state.resnet50 = None
#%%
######################################################################################################################################################################
# Setting up the Options
######################################################################################################################################################################

app_mode = st.sidebar.selectbox("Please select from the following:", ["Overview", "LIME" , "SHAP"])

if app_mode == "Overview":

    st.write("""
                ######  
                """)
        
    expander_model_img = st.expander("Model Overview of STimage:")
    expander_model_img.image(Image.open(Path+'STimage-web_app_image1.png'), caption='Model Architecture',width=600)

    expander_norm = st.expander("More about Normalisation of STimage:")
    expander_norm.image(Image.open(Path+'/Stimage_web_app_data_norm.png'), caption='Data Augmentation and Normalisation',width=600)

    st.sidebar.write(" ------ ")
    st.sidebar.info("Enter config file for generating normalised tiles and predicting gene expression!")

    #st.write("""###### """)    
    config_file = st.file_uploader("Input the config file to normalise image tiles", type="ini")
    
    ######################################################################################################################################################################
    # Preprocessing and tiling
    ######################################################################################################################################################################
    if config_file is not None:
        import Preprocessing
    
    
    st.write("");st.write("");st.write("")
    expander_team_img = st.expander("STimage Team:")
    expander_team_img.image(Image.open(Path+'Meet_the_team.png'), caption='STimage Team',width=600)    
    expander_team_img.write("Please contact Dr Quan Nguyen , Xiao Tan or Onkar Mulay!")
    expander_team_img.write('quan.nguyen@uq.edu.au')
    expander_team_img.write('xiao.tan@uq.edu.au')
    expander_team_img.write('o.mulay@uq.net.au')
    
#%%
######################################################################################################################################################################
# Perform LIME
######################################################################################################################################################################

elif app_mode =="LIME":

    st.sidebar.warning("Upload Image Tile, Trained Model and Enter Gene Names:")
    expander_lime = st.expander('LIME Model:')
    expander_lime.image(Image.open(Path+'/Lime_Classification.png'), caption='LIME model',width=600)

    gene_list = st.text_input("Space separated Gene Names").split() #['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']

    st.markdown('<h5 style="color: black;"> Uploading the Image Tiles',unsafe_allow_html=True)
    uploaded_image_file = st.file_uploader("Select image tile:")
    if uploaded_image_file is not None:
        image1 = uploaded_image_file
        
        col_1_lime, col_2_lime = st.columns(2)
        with col_1_lime:
            st.image(Image.open(image1))
        
        val = st.slider('Set the threshold:', 0.25, 1.0, 0.7)
        with col_2_lime:
            st.image(watershed_segment(np.asarray(Image.open(image1))),clamp=True)


        agree = st.checkbox('Run Classification LIME')
        if agree:
              
            st.markdown('<h5 style="color: black;"> LIME Classification Model', unsafe_allow_html=True)
            resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")
    
            filename = st.file_uploader('Enter the path of saved model:') #,accept_multiple_files=True,type=["csv","pkl"]
            if filename is not None:
                model_c = joblib.load(filename)
            
                col4, col5 = st.columns(2)
                with col4:
                    gene_input_1 = gene_list[0] 
                    
                with col5:
                    gene_input_2 = gene_list[1] 

                col6, col7 = st.columns(2)            
                with col6:
                    st.write(gene_input_1)
                    heatmap1 = LIME(image1,gene_input_1)
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(image1))
                    plt.imshow(heatmap1, alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap1.max(), vmax = heatmap1.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
               
                with col7:
                    st.write(gene_input_2)
                    heatmap2 = LIME(image1,gene_input_2)
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(image1))
                    plt.imshow(heatmap2, alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap2.max(), vmax = heatmap2.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()

                st.success("Gene Specific Nuclei were computed by LIME")    
                
                st.session_state.image_var = image1
                st.session_state.gene_1 = gene_input_1
                st.session_state.gene_2 = gene_input_2
                st.session_state.gene_list = gene_list
                st.session_state.model_clf = model_c
                st.session_state.resnet50 = resnet_model

#%%
        agree2 = st.checkbox('Run Regression LIME')
        if agree2:
            
            st.markdown('<h5 style="color: black;"> LIME for Regression Model', unsafe_allow_html=True)
                     

            filename = st.file_uploader('TF.Keras model file (.h5py.zip)', type='zip')#st.text_input('Enter the path of saved model weights:')
            if filename is not None:
                #st.warning("Enter h5-File Path")
    
                myzipfile = zipfile.ZipFile(filename)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    myzipfile.extractall(tmp_dir)
                    root_folder = myzipfile.namelist()[0] 
                    model_dir = os.path.join(tmp_dir, root_folder)
                    #model = tf.keras.models.load_model(model_dir)
    
                    model_weights = model_dir
                    model = CNN_NB_multiple_genes((299, 299, 3), 1522)
                    model.load_weights(model_weights)
                    model.compile(loss=negative_binomial_loss,  
                                optimizer=tf.keras.optimizers.Adam(0.0001))
                                    
                    col8, col9 = st.columns(2)                
                    with col8:
                        gene_input_1 = gene_list[0] 
                        
                    with col9:
                        gene_input_2 = gene_list[1] 
    
                    col10, col11 = st.columns(2)         
    
                    with col10:
                        st.write(gene_input_1)
                        heatmap1 = LIME_reg(image1,gene_input_1)
                        fig, ax = plt.subplots()
                        plt.imshow(Image.open(image1))
                        plt.imshow(heatmap1, alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap1.max(), vmax = heatmap1.max())
                        st.write(fig)
                        plt.colorbar()
                        plt.show()
                    
                    with col11:
                        st.write(gene_input_2)
                        heatmap2 = LIME_reg(image1,gene_input_2)
                        fig, ax = plt.subplots()
                        plt.imshow(Image.open(image1))
                        plt.imshow(heatmap2, alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap2.max(), vmax = heatmap2.max())
                        st.write(fig)
                        plt.colorbar()
                        plt.show()
                        
                    st.success("Gene Specific Nuclei were computed by LIME")

                    st.session_state.model_reg = model                    
            

#%%
#%%
######################################################################################################################################################################
# Perform SHAP
######################################################################################################################################################################

elif app_mode =="SHAP":
    
    image1 = Image.open(st.session_state.image_var)
    segments_slic = watershed_segment(image1)
    image_orig = img_to_array(image1)
    
    gene_list = st.session_state.gene_list
    gene_input_1 = st.session_state.gene_1
    gene_input_2 = st.session_state.gene_2
    resnet_model = st.session_state.resnet50
    model_c = st.session_state.model_clf
    model = st.session_state.model_reg



    agree3 = st.checkbox('Run Classification SHAP',value=False)
    if agree3:
        st.markdown('<h5 style="color: black;"> SHAP for Regression Model', unsafe_allow_html=True)
        heatmap1 = SHAP_Classification(segments_slic,image_orig,gene_input_1)
        heatmap2 = SHAP_Classification(segments_slic,image_orig,gene_input_2)
                
        col12, col13 = st.columns(2) 
        with col12:
            st.write(gene_input_1)
            fig, ax = plt.subplots()
            plt.imshow(Image.open(st.session_state.image_var))
            plt.imshow(heatmap1[0], alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap1[0].max(), vmax = heatmap1[0].max())
            st.write(fig)
            plt.colorbar()
            plt.show()
            
        with col13:
            st.write(gene_input_2)
            fig, ax = plt.subplots()
            plt.imshow(Image.open(st.session_state.image_var))
            plt.imshow(heatmap2[0], alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap1[0].max(), vmax = heatmap1[0].max())
            st.write(fig)
            plt.colorbar()
            plt.show()
            
        st.success("Gene Specific Nuclei were computed by SHAP")
    
    
    agree4 = st.checkbox('Run Regression SHAP',value=False)
    if agree4:
        heatmap3 = SHAP_reg(segments_slic,image_orig,gene_input_1)
        i = gene_list.index(gene_input_2)
        heatmap4 = SHAP_reg(segments_slic,image_orig,gene_input_2)
                
        col12, col13 = st.columns(2) 
        with col12:
            st.write(gene_input_1)
            fig, ax = plt.subplots()
            plt.imshow(Image.open(st.session_state.image_var))
            plt.imshow(heatmap3[0], alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap3[0].max(), vmax = heatmap3[0].max())
            st.write(fig)
            plt.colorbar()
            plt.show()
            
        with col13:
            st.write(gene_input_2)
            fig, ax = plt.subplots()
            plt.imshow(Image.open(st.session_state.image_var))
            plt.imshow(heatmap4[0], alpha = 0.45, cmap = 'RdYlBu_r', vmin  = -heatmap4[0].max(), vmax = heatmap4[0].max())
            st.write(fig)
            plt.colorbar()
            plt.show()
        
        st.success("Gene Specific Nuclei were computed by SHAP")


