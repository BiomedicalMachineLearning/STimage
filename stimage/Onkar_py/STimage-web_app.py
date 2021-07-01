import streamlit as st; import streamlit.components.v1 as components
import os; from os import listdir; import pandas as pd; import numpy as np; from numpy import asarray
import h5py; from pathlib import Path; import pickle; import tensorflow as tf; import copy; import warnings
import PIL; from PIL import Image; PIL.Image.MAX_IMAGE_PIXELS = 933120000


import shap; shap.initjs(); import lime; from lime import lime_image; import lime.lime_tabular
import stlearn; from tqdm import tqdm; import math
import plotly.express as px; import matplotlib.pyplot as plt; from matplotlib import cm as cm
import cv2; import joblib; from sklearn.metrics import confusion_matrix; import seaborn as sns


from tensorflow import keras; from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.applications import VGG16, ResNet50, inception_v3, DenseNet121
from keras.applications import imagenet_utils; from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image as load_img; from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Input, Lambda
from tensorflow.keras.preprocessing import image as image_fun
from keras.applications.resnet50 import ResNet50, preprocess_input as pi;


from sklearn.preprocessing import LabelEncoder; from sklearn import preprocessing; from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score; from sklearn.preprocessing import LabelEncoder; from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier; from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb; 


import scipy as sp; from scipy import ndimage as ndi
from skimage.feature import peak_local_max; from skimage.segmentation import watershed; from skimage.measure import label
import skimage; from skimage.color import rgb2hed; from skimage.morphology import area_opening
#%%
#D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/CNN_NB_8genes_model.h5

#wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"
#%%
st.set_page_config(
    page_title="STimage",
    layout="centered",
    initial_sidebar_state="auto",
)

st.sidebar.header("""Navigate to Algorihms""")
rad = st.sidebar.radio("Choose the list", ["Data - Exploration", "LIME Plots for Gene Expression Classification", "LIME Plots for Gene Expression Prediction"])

title_container = st.beta_container()
col1, mid, col2 = st.beta_columns([3, 1, 4])


with title_container:
        with col2:
            st.markdown('<h1 style="color: purple;">STimage</h2>',
                            unsafe_allow_html=True)
#%%
if rad =="Data - Exploration":

    st.write("""
                 #### This project aims at developing various computational models to integrate histopathological images with spatial transcriptomics data to predict the spatial gene expression patterns as hallmarks for the detection and classification of cancer cells in a tissue. The success of this project would contribute to digital pathological diagnosis and thus uncover the spatial dominance of one gene over the other (if it exists) in order to understand the difference between the expression of the genes in cancer and non-cancer regions of the tissue. 
                 """)
    st.write("""""")
    st.write("""### Explore Image Tiles""")    

    img = st.file_uploader("Image Tile Upload", type="jpeg")
    if img is not None:
        col1, col2, col3 = st.beta_columns([2,6,1])
        img = Image.open(img)
        with col1:
            st.write("")
        with col2:
            st.image(img)
        with col3:
            st.write("")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        col1, col2, col3 = st.beta_columns(3)
        def getRed(redVal):
            return '#%02x%02x%02x' % (redVal, 0, 0)
    
        def getGreen(greenVal):
            return '#%02x%02x%02x' % (0, greenVal, 0)
        
        def getBlue(blueVal):
            return '#%02x%02x%02x' % (0, 0, blueVal)
        histogram = img.histogram()
        l1 = histogram[0:256]    
        l2 = histogram[256:512]    
        l3 = histogram[512:768]
        
        with col1:
            fig, ax = plt.subplots()
            for i in range(0, 256):    
                plt.bar(i, l1[i], color = getRed(i), edgecolor=getRed(i), alpha=0.3)
            st.pyplot()
        with col2:
            fig, ax = plt.subplots()
            plt.figure(1)
            for i in range(0, 256):    
                plt.bar(i, l2[i], color = getGreen(i), edgecolor=getGreen(i),alpha=0.3)
            st.pyplot()
        with col3:
            fig, ax = plt.subplots()
            plt.figure(2)
            for i in range(0, 256):    
                plt.bar(i, l3[i], color = getBlue(i), edgecolor=getBlue(i),alpha=0.3)
            st.pyplot()

#%%
if rad =="LIME Plots for Gene Expression Classification":
    
    def model_predict_gene(gene):
                    i = gene_list.index(gene)
                    def combine_model_predict(tile):
                        feature = resnet_model.predict(tile)
                        feature = feature.reshape((10, 2048))
                        prediction = Model_LGBM.predict_proba(feature)
                        return prediction[i]
                    return combine_model_predict
                
    def transform_img_fn(im):
        out = []
        img = image_fun.load_img(im, target_size=(299, 299))
        x = image_fun.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
        return np.vstack(out)
    
    def watershed_segment(image):
        annotation_hed = rgb2hed(image)
        annotation_h = annotation_hed[:,:,0]
        annotation_h *= 255.0 / np.percentile(annotation_h, q=80)
        thresh = skimage.filters.threshold_otsu(annotation_h)
        im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
            annotation_h < thresh
        )
        distance = ndi.distance_transform_edt(im_fgnd_mask)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=im_fgnd_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(annotation_h, markers, mask=im_fgnd_mask)
        im_nuclei_seg_mask = area_opening(labels, area_threshold=80).astype(np.int)
        return im_nuclei_seg_mask
    
    st.sidebar.subheader("Options")
    resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")
    
    checkbox1 = st.sidebar.checkbox("Biomarkers Pre-Trained")
    if checkbox1:
        Top_500genes_train = ['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
        st.subheader("List of Genes")
        st.write(Top_500genes_train)
        
        uploaded_image_file = st.file_uploader("Image Tile Upload", type="jpeg")
        if uploaded_image_file is not None:
            im = uploaded_image_file
            Image_tile = Image.open(im)
            st.image(Image_tile)
        
            import os
            filename = st.file_uploader('Enter the path of saved model:')
            if filename is not None:
                Model_LGBM = joblib.load(filename)
            
                
               
                gene_list = Top_500genes_train
                col15, col16 = st.beta_columns(2)
                with col15:
                    user_input = st.text_input("Enter Gene Name:", 'COX6C')
                   
                with col16:
                    user_input_2 = st.text_input("Enter Gene Name:", 'MALAT1')
                col1, col2, col3 = st.beta_columns(3)
                
                st.markdown("""
                            #### Quickshift Segmentation for LIME
                         """)
                
                        
                images = transform_img_fn(im)
                explainer = lime_image.LimeImageExplainer()
                explanation_quick_us1 = explainer.explain_instance(images[0].astype('double'), 
                                                     model_predict_gene(user_input), 
                                                     top_labels=3, num_samples=100,
                                                     segmentation_fn=None)
                explanation_watershed_us1 = explainer.explain_instance(images[0].astype('double'), 
                                             model_predict_gene(user_input), 
                                             top_labels=3, num_samples=100,
                                             segmentation_fn=watershed_segment)
                
                explanation_quick_us2 = explainer.explain_instance(images[0].astype('double'), 
                                                     model_predict_gene(user_input_2), 
                                                     top_labels=3, num_samples=100,
                                                     segmentation_fn=None)
                explanation_watershed_us2 = explainer.explain_instance(images[0].astype('double'), 
                                             model_predict_gene(user_input_2), 
                                             top_labels=3, num_samples=100,
                                             segmentation_fn=watershed_segment)
                
                with col1:
            
            
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu',vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col2:
            
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy())#.astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col3:
                    
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy())#.astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                st.markdown("""
                        #### Watershed Segmentation for LIME
                     """)
            
                col7, col8, col9 = st.beta_columns(3)
            
                with col7:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col8:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col9:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
            
                st.markdown("""
                        #### Quickshift Segmentation for LIME
                     """)
                    
                col4, col5, col6 = st.beta_columns(3)
                
                with col4:
                    
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col5:
            
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
            
                with col6:
                    
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
               
                st.markdown("""
                        #### Watershed Segmentation for LIME
                     """)
                    
                col10, col11, col12 = st.beta_columns(3)
                
                with col10:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col11:
            
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col12:
            
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
    checkbox2 = st.sidebar.checkbox("Custom Input")
    if checkbox2: 
        Top_500genes_train = st.text_input("Gene Name").split()
        st.subheader("List of Genes")
        st.write(Top_500genes_train)
        
        uploaded_image_file = st.file_uploader("Image Tile Upload", type="jpeg")
        if uploaded_image_file is not None:
            im = uploaded_image_file
            Image_tile = Image.open(im)
            st.image(Image_tile)
        
            import os
            filename = st.file_uploader('Enter the path of saved model:')
            if filename is not None:
                Model_LGBM = joblib.load(filename)
               
                gene_list = Top_500genes_train
                col15, col16 = st.beta_columns(2)
                with col15:
                    user_input = st.text_input("Enter Gene Name:", 'COX6C')
                   
                with col16:
                    user_input_2 = st.text_input("Enter Gene Name:", 'MALAT1')
                col1, col2, col3 = st.beta_columns(3)
                
                st.markdown("""
                            #### Quickshift Segmentation for LIME
                         """)
                
                        
                images = transform_img_fn(im)
                explainer = lime_image.LimeImageExplainer()
                explanation_quick_us1 = explainer.explain_instance(images[0].astype('double'), 
                                                     model_predict_gene(user_input), 
                                                     top_labels=3, num_samples=100,
                                                     segmentation_fn=None)
                explanation_watershed_us1 = explainer.explain_instance(images[0].astype('double'), 
                                             model_predict_gene(user_input), 
                                             top_labels=3, num_samples=100,
                                             segmentation_fn=watershed_segment)
                
                explanation_quick_us2 = explainer.explain_instance(images[0].astype('double'), 
                                                     model_predict_gene(user_input_2), 
                                                     top_labels=3, num_samples=100,
                                                     segmentation_fn=None)
                explanation_watershed_us2 = explainer.explain_instance(images[0].astype('double'), 
                                             model_predict_gene(user_input_2), 
                                             top_labels=3, num_samples=100,
                                             segmentation_fn=watershed_segment)
                
                with col1:
            
            
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu',vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col2:
            
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy())#.astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col3:
                    
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy())#.astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                st.markdown("""
                        #### Watershed Segmentation for LIME
                     """)
            
                col7, col8, col9 = st.beta_columns(3)
            
                with col7:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col8:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col9:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
            
                st.markdown("""
                        #### Quickshift Segmentation for LIME
                     """)
                    
                col4, col5, col6 = st.beta_columns(3)
                
                with col4:
                    
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col5:
            
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
            
                with col6:
                    
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
               
                st.markdown("""
                        #### Watershed Segmentation for LIME
                     """)
                    
                col10, col11, col12 = st.beta_columns(3)
                
                with col10:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col11:
            
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[1]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                with col12:
            
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[2]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
#%%
if rad =="LIME Plots for Gene Expression Prediction":

    st.sidebar.subheader("Options")
    class PrinterCallback(tf.keras.callbacks.Callback):
                
        def on_epoch_end(self, epoch, logs=None):
            print('EPOCH: {}, Train Loss: {}, Val Loss: {}'.format(epoch,
                                                                   logs['loss'],
                                                                   logs['val_loss']))
    
        def on_epoch_begin(self, epoch, logs=None):
            print('-' * 50)
            print('STARTING EPOCH: {}'.format(epoch))
    
    
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
    
    def transform_img_fn(im):
        out = []
        img = image_fun.load_img(im, target_size=(299, 299))
        x = image_fun.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
        return np.vstack(out)
    
    def watershed_segment(image):
        annotation_hed = rgb2hed(image)
        annotation_h = annotation_hed[:,:,0]
        annotation_h *= 255.0 / np.percentile(annotation_h, q=80)
    #     annotation_h = np.clip(annotation_h, a_min=0, a_max=255)
        thresh = skimage.filters.threshold_otsu(annotation_h)
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
    
    

    checkbox3 = st.sidebar.checkbox("Biomarkers")
    if checkbox3:

        uploaded_image_file = st.file_uploader("Image Tile Upload", type="jpeg")
        if uploaded_image_file is not None:
            im = uploaded_image_file
            Image_tile = Image.open(im)
            st.image(Image_tile)            

            Top_500genes_train = ['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']  
            st.subheader("List of Genes")
            st.write(Top_500genes_train)

            filename = st.text_input('Enter the path of saved model weights:')
            if filename is None:
                st.warning("Enter h5-File Path")
            
            else:
                model_weights = filename
                
                model = CNN_NB_multiple_genes((299, 299, 3), 8)
                model.load_weights(model_weights)
                model.compile(loss=negative_binomial_loss,
                              optimizer=tf.keras.optimizers.Adam(0.0001))
    
                gene_list = Top_500genes_train
                col15, col16 = st.beta_columns(2)
                with col15:
                    user_input = st.text_input("Enter Gene Name:", 'COX6C')
                   
                with col16:
                    user_input_2 = st.text_input("Enter Gene Name:", 'MALAT1')
         
                images = transform_img_fn(im)
                explainer = lime_image.LimeImageExplainer()
                explanation_quick_us1 = explainer.explain_instance(images[0].astype('double'), 
                                                     model_predict_gene(user_input), 
                                                     top_labels=3, num_samples=100,
                                                     segmentation_fn=None)
                
                explanation_quick_us2 = explainer.explain_instance(images[0].astype('double'), 
                                                     model_predict_gene(user_input_2), 
                                                     top_labels=3, num_samples=100,
                                                     segmentation_fn=None)
                
                st.markdown("""
                        #### Quickshift Segmentation for LIME
                     """)
                col1, col2 = st.beta_columns(2)
                
               
            
                with col1:
            
                    dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy())#.astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
     
                with col2:
            
                    dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                    
                st.markdown("""
                        #### Watershed Segmentation for LIME
                     """)
                     
                col4, col5 = st.beta_columns(2)
                
                
                     
                explanation_watershed_us1 = explainer.explain_instance(images[0].astype('double'), 
                                             model_predict_gene(user_input), 
                                             top_labels=3, num_samples=100,
                                             segmentation_fn=watershed_segment)
                
                explanation_watershed_us2 = explainer.explain_instance(images[0].astype('double'), 
                                             model_predict_gene(user_input_2), 
                                             top_labels=3, num_samples=100,
                                             segmentation_fn=watershed_segment)
                     
                with col4:
                    
                    dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                
                with col5:
            
                    dict_heatmap = dict(explanation_watershed_us2.local_exp[explanation_watershed_us2.top_labels[0]])
                    heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                    fig, ax = plt.subplots()
                    plt.imshow(Image.open(im))#.numpy().astype(int))
                    plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                    st.write(fig)
                    plt.colorbar()
                    plt.show()
                
    checkbox4 = st.sidebar.checkbox("Custom Input")
    if checkbox4:
        
        uploaded_image_file = st.file_uploader("Image Tile Upload", type="jpeg")
        if uploaded_image_file is not None:
            im = uploaded_image_file
            Image_tile = Image.open(im)
            st.image(Image_tile)
        
            Top_500genes_train = st.text_input("Gene Name").split()
            st.subheader("List of Genes")
            st.write(Top_500genes_train)
            
            if Top_500genes_train is not None:
                filename = st.text_input('Enter the path of saved model weights:')
                if filename is None:
                    model_weights = filename
                    
                    model = CNN_NB_multiple_genes((299, 299, 3), 8)
                    model.load_weights(model_weights)
                    model.compile(loss=negative_binomial_loss,
                                  optimizer=tf.keras.optimizers.Adam(0.0001))
        
                    gene_list = Top_500genes_train
                    col15, col16 = st.beta_columns(2)
                    with col15:
                        user_input = st.text_input("Enter Gene Name:", Top_500genes_train[0])
                       
                    with col16:
                        user_input_2 = st.text_input("Enter Gene Name:", Top_500genes_train[1])
             
                    images = transform_img_fn(im)
                    explainer = lime_image.LimeImageExplainer()
                    explanation_quick_us1 = explainer.explain_instance(images[0].astype('double'), 
                                                         model_predict_gene(user_input), 
                                                         top_labels=3, num_samples=100,
                                                         segmentation_fn=None)
                    
                    explanation_quick_us2 = explainer.explain_instance(images[0].astype('double'), 
                                                         model_predict_gene(user_input_2), 
                                                         top_labels=3, num_samples=100,
                                                         segmentation_fn=None)
                    
                    st.markdown("""
                            #### Quickshift Segmentation for LIME
                         """)
                    col1, col2 = st.beta_columns(2)
                    
                   
                
                    with col1:
                
                        dict_heatmap = dict(explanation_quick_us1.local_exp[explanation_quick_us1.top_labels[0]])
                        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us1.segments) 
                        fig, ax = plt.subplots()
                        plt.imshow(Image.open(im))#.numpy())#.astype(int))
                        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                        st.write(fig)
                        plt.colorbar()
                        plt.show()
         
                    with col2:
                
                        dict_heatmap = dict(explanation_quick_us2.local_exp[explanation_quick_us2.top_labels[0]])
                        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick_us2.segments) 
                        fig, ax = plt.subplots()
                        plt.imshow(Image.open(im))#.numpy().astype(int))
                        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                        st.write(fig)
                        plt.colorbar()
                        plt.show()
                        
                    st.markdown("""
                            #### Watershed Segmentation for LIME
                         """)
                         
                    col4, col5 = st.beta_columns(2)
                    
                    
                         
                    explanation_watershed_us1 = explainer.explain_instance(images[0].astype('double'), 
                                                 model_predict_gene(user_input), 
                                                 top_labels=3, num_samples=100,
                                                 segmentation_fn=watershed_segment)
                    
                    explanation_watershed_us2 = explainer.explain_instance(images[0].astype('double'), 
                                                 model_predict_gene(user_input_2), 
                                                 top_labels=3, num_samples=100,
                                                 segmentation_fn=watershed_segment)
                         
                    with col4:
                        
                        dict_heatmap = dict(explanation_watershed_us1.local_exp[explanation_watershed_us1.top_labels[0]])
                        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us1.segments) 
                        fig, ax = plt.subplots()
                        plt.imshow(Image.open(im))#.numpy().astype(int))
                        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                        st.write(fig)
                        plt.colorbar()
                        plt.show()
                    
                    with col5:
                
                        dict_heatmap = dict(explanation_watershed_us2.local_exp[explanation_watershed_us2.top_labels[0]])
                        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed_us2.segments) 
                        fig, ax = plt.subplots()
                        plt.imshow(Image.open(im))#.numpy().astype(int))
                        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
                        st.write(fig)
                        plt.colorbar()
                        plt.show()
#%%