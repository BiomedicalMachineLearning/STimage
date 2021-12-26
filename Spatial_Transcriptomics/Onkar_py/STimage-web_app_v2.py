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



from sklearn.preprocessing import LabelEncoder; from sklearn import preprocessing; from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score; from sklearn.preprocessing import LabelEncoder; from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier; from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import AgglomerativeClustering
import lightgbm as lgb; 


import scipy as sp; from scipy import ndimage as ndi; import time
from skimage.feature import peak_local_max; from skimage.segmentation import watershed; from skimage.measure import label
import skimage; from skimage.color import rgb2hed; from skimage.morphology import area_opening; from skimage.segmentation import mark_boundaries

import zipfile
import tempfile
#%%
#D:/Onkar_D/UQ/Project_Spt.Transcriptomics/Output_files/CNN_NB_8genes_model.h5

#wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"
#%%
st.set_page_config(
    page_title="STimage Web-App",
    layout="centered",
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

st.sidebar.header("""Navigate to Algorihms""")
rad = st.sidebar.selectbox("Choose the list", ["Data - Exploration", "LIME for Classification", "LIME for Regression"])


title_container = st.beta_container()
col1, mid, col2 = st.beta_columns([2, 2, 2])


with title_container:
        with mid:
            st.markdown('<h1 style="color: purple;">STimage Web-App',
                            unsafe_allow_html=True)
st.markdown('<h2 style="color: grey;">Predicting Spatial Gene Expression Using Tissue Morphology and Spatial Transcriptomics',
                            unsafe_allow_html=True)

#%%
if rad =="Data - Exploration":

    model_arch = Image.open('D:/Onkar_D/UQ/Project_Spt.Transcriptomics/Stimage_web_app_model_arch.png')
    st.image(model_arch, caption='Model Architecture',width=600)

    st.write("""
                 ### This project aims at developing a deep learning based computational model to integrate histopathological images with spatial transcriptomics data to predict the spatial gene expression patterns as hallmarks for the detection and classification of cancer cells in a tissue.  
                 """)
    st.write("""""")
    
    data_norm_img = Image.open('D:/Onkar_D/UQ/Project_Spt.Transcriptomics/Stimage_web_app_data_norm.png')
    st.image(data_norm_img, caption='Data Augmentation and Normalisation',width=600)
    
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
if rad =="LIME for Classification":
    
    @st.cache
    def model_predict_gene(gene):
        i = gene_list.index(gene)
        def combine_model_predict(tile1):
            feature1 = resnet_model.predict(tile1)
            prediction = model_c.predict_proba(feature1)#[0]
            return prediction[i]#.reshape(-1,1)
        return combine_model_predict
    
    @st.cache
    def model_predict_2gene(gene, tile2):
        i = gene_list.index(gene)
    
        def combine_model_predict2(tile1):
            feature1 = resnet_model.predict(tile1)
            feature2 = resnet_model.predict(tile2)
            feature = np.concatenate((feature1, feature2), axis=1)
            prediction = model_c.predict_proba(feature)
            return prediction[i]
        return combine_model_predict2
    
    @st.cache
    def transform_img_fn(img):
        out = []
        x = image_fun.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
        return np.vstack(out)

    @st.cache
    def watershed_segment(image):
        annotation_hed = rgb2hed(image)
        annotation_h = annotation_hed[:,:,0]
        annotation_h *= 255.0 / np.percentile(annotation_h, q=80)
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
    
    st.sidebar.subheader("Options")
    resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")
    
    checkbox1 = st.sidebar.checkbox("Biomarkers")
    small_tile=0; big_tile=1;
    if checkbox1:
        Top_500genes_train = ['COX6C', 'CD74'] #['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
        st.subheader("List of Genes")
        st.write(Top_500genes_train)
        st.subheader("Uploading the Image Tiles")
        

        uploaded_image_file1 = st.file_uploader("Enter the path of the image tile:")
        if uploaded_image_file1 is not None:
            image1 = uploaded_image_file1
            Image_tile1 = Image.open(image1)#Image.open
            st.image(Image_tile1)
            image1_trans = transform_img_fn(Image_tile1)
        

        filename = st.file_uploader('Enter the path of saved model:')
        if filename is not None:
            model_c = joblib.load(filename)
 
            
            gene_list = Top_500genes_train
            col15, col16 = st.beta_columns(2)
            with col15:
                user_input = st.text_input("Enter Gene Name:", 'COX6C')
               
            with col16:
                user_input_2 = st.text_input("Enter Gene Name:", 'CD74')
            
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            my_bar.empty()
            st.info("Gene Specific Nuclei are being computed by LIME")
            
            
            explainer = lime_image.LimeImageExplainer()
            
            
            explanation_gene1 = explainer.explain_instance(image1_trans[0].astype('double'), model_predict_gene(user_input), top_labels=2, hide_color=0, num_samples=100, segmentation_fn=watershed_segment)
            explanation_gene2 = explainer.explain_instance(image1_trans[0].astype('double'), model_predict_gene(user_input_2), top_labels=2, hide_color=0, num_samples=100, segmentation_fn=watershed_segment)

                
            st.markdown("""
                    #### Watershed Segmentation for LIME
                 """)
        
            col7, col8 = st.beta_columns(2)
        
            with col7:
                
                temp, mask = explanation_gene1.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
                fig, ax = plt.subplots()
                plt.imshow(mark_boundaries(temp, mask).astype('uint8'))
                st.write(fig)
                plt.show()
                
            with col8:
                
                temp, mask = explanation_gene2.get_image_and_mask(1, positive_only=False, num_features=100, hide_rest=True)
                fig, ax = plt.subplots()
                plt.imshow(mark_boundaries(temp, mask).astype('uint8'))
                st.write(fig)
                plt.show()
        

    checkbox2 = st.sidebar.checkbox("Custom Input") #
    if checkbox2: 
        small_tile=0; big_tile=1;
        Top_500genes_train = st.text_input("Gene Name").split() #['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
        st.subheader("List of Genes")

        
        st.subheader("Uploading the Image Tiles")
        col_im1 = st.beta_columns(1)
        with col_im1:
            uploaded_image_file1 = st.file_uploader("Small Image Tile Upload", type="jpeg", key=small_tile)
            if uploaded_image_file1 is not None:
                image1 = uploaded_image_file1
                Image_tile1 = Image.open(image1)
                st.image(Image_tile1)
                image1_trans = transform_img_fn(Image_tile1)
        

        filename = st.file_uploader('Enter the path of saved model:')
        if filename is not None:
            model_c = joblib.load(filename)
            
                
            gene_list = Top_500genes_train
            col15, col16 = st.beta_columns(2)
            with col15:
                user_input = st.text_input("Enter Gene Name:", 'COX6C')
               
            with col16:
                user_input_2 = st.text_input("Enter Gene Name:", 'CD74')
            
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1)
            my_bar.empty()
            st.info("Gene Specific Nuclei are being computed by LIME")
            
            
            explainer = lime_image.LimeImageExplainer()
            
            explanation_gene1 = explainer.explain_instance(image1_trans[0].astype('double'), model_predict_gene(user_input), top_labels=2, hide_color=0, num_samples=1000, segmentation_fn=watershed_segment)
            explanation_gene2 = explainer.explain_instance(image1_trans[0].astype('double'), model_predict_gene(user_input_2), top_labels=2, hide_color=0, num_samples=1000, segmentation_fn=watershed_segment)

                
            st.markdown("""
                    #### Watershed Segmentation for LIME
                 """)
        
            col7, col8 = st.beta_columns(2)
        
            with col7:
                
                temp, mask = explanation_gene1.get_image_and_mask(1, positive_only=False, num_features=1000, hide_rest=True)
                fig, ax = plt.subplots()
                plt.imshow(mark_boundaries(temp, mask).astype('uint8'))
                st.write(fig)
                plt.show()
                
            with col8:
                
                temp, mask = explanation_gene2.get_image_and_mask(1, positive_only=False, num_features=1000, hide_rest=True)
                fig, ax = plt.subplots()
                plt.imshow(mark_boundaries(temp, mask).astype('uint8'))
                st.write(fig)
                plt.show()
   
#%%
if rad =="LIME for Regression":

    st.sidebar.subheader("Options")
    class PrinterCallback(tf.keras.callbacks.Callback):
                
        def on_epoch_end(self, epoch, logs=None):
            print('EPOCH: {}, Train Loss: {}, Val Loss: {}'.format(epoch,
                                                                   logs['loss'],
                                                                   logs['val_loss']))
    
        def on_epoch_begin(self, epoch, logs=None):
            print('-' * 50)
            print('STARTING EPOCH: {}'.format(epoch))
    
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
    
    @st.cache
    def transform_img_fn(img):
        out = []
        x = image_fun.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
        return np.vstack(out)
    
    @st.cache
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
        
                    gene_list = Top_500genes_train
                    col15, col16 = st.beta_columns(2)
                    with col15:
                        user_input = st.text_input("Enter Gene Name:", 'COX6C')
                    
                    with col16:
                        user_input_2 = st.text_input("Enter Gene Name:", 'MALAT1')

                    my_bar_r = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                    my_bar_r.progress(percent_complete + 1)
                    my_bar_r.empty()
                    st.info("Gene Specific Nuclei are being computed by LIME")

            
                    images = transform_img_fn(Image_tile)
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
             
                    images = transform_img_fn(Image_tile)
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
