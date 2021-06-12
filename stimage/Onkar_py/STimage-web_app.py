#%%
import streamlit as st; import pandas as pd; import numpy as np
import pickle; 
from sklearn.ensemble import RandomForestClassifier; from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb; import skimage
from skimage.color import rgb2hed; import seaborn as sns
from skimage.feature import peak_local_max; from skimage.segmentation import watershed; from skimage.measure import label
import scipy as sp
from sklearn.metrics import plot_confusion_matrix
from scipy import ndimage as ndi
from skimage.morphology import area_opening
import math
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder; 
from sklearn import preprocessing; from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score; from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder; from sklearn.model_selection import train_test_split
from sklearn import preprocessing; from sklearn.linear_model import LogisticRegression
import shap; import numpy as np; shap.initjs()
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
from matplotlib import cm as cm
import glob; from PIL import Image
import os; import sys; import pandas as pd; import numpy as np; from numpy import array; from numpy import argmax
import lime
from tensorflow import keras
import lime.lime_tabular
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import to_categorical; from sklearn.preprocessing import LabelEncoder; from sklearn.preprocessing import OneHotEncoder; from sklearn.model_selection import train_test_split
import stlearn
stlearn.settings.set_figure_params(dpi=300)
from pathlib import Path
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from anndata import AnnData
import pandas as pd
from typing import Optional, Union
from anndata import AnnData
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf; from matplotlib import pyplot as plt
from keras import backend; from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential; from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD; from keras.models import Model; from tensorflow.keras import regularizers; from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input as pi; from keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K; from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions; import cv2
import lime; from lime import lime_image
from keras.preprocessing import image as image_to_load; import numpy as np; import h5py; from keras.models import load_model 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray
from os import listdir
from sklearn.cluster import AgglomerativeClustering
import os; import pandas as pd; import numpy as np
import PIL; from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16, ResNet50, inception_v3, DenseNet121
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import warnings; from tensorflow.keras.preprocessing import image as image_fun
import tensorflow as tf; import copy
#%%
BASE_PATH = Path("D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files")
SAMPLE = ['block1', 'block2']
count_file = ['V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5','V1_Breast_Cancer_Block_A_Section_2_filtered_feature_bc_matrix.h5']
main_image = ["V1_Breast_Cancer_Block_A_Section_1_image.tif", "V1_Breast_Cancer_Block_A_Section_2_image.tif"]
AnnData = []
for i in range(0,len(SAMPLE)):
    Samples = stlearn.Read10X(BASE_PATH / SAMPLE[i], 
                      library_id=SAMPLE[i], 
                      count_file=count_file[i],
                      quality="fulres",)
    img = plt.imread(BASE_PATH / SAMPLE[i] /main_image[i], 0)
    Samples.uns["spatial"][SAMPLE[i]]['images']["fulres"] = img
    AnnData.append(Samples)

Sample1_un_norm, Sample2_un_norm = copy.copy(AnnData[0]), copy.copy(AnnData[1])

for i in range(0,len(AnnData)):
    stlearn.pp.normalize_total(AnnData[i])
#%%
#ResNet-50 Features
#Cancer-vs-Non_Cancer Spots 
wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"
Cluster_train = pd.read_csv(wd+'Cluster_train.csv').iloc[:,1:]
Cluster_test = pd.read_csv(wd+'Cluster_test.csv').iloc[:,1:]
Model_LGBM = joblib.load(wd+'ResNet50-LGBM_200_unnorm_updated.pkl')
model_weights = wd+'CNN_NB_8genes_model.h5'
#%%
#st.set_page_config(layout="wide")
icon = Image.open(wd+'Logo.png')

st.set_page_config(
    page_title="STimage",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="auto",
)

st.sidebar.header("""Navigate to Algorihms""")
rad = st.sidebar.radio("Choose the list", ["Data - Exploration", "Cancer vs Non-Cancer Prediction from Biomarker Expression", "AUROC-Curve",  "LIME Plots for Gene Expression Classification", "LIME Plots for Gene Expression Prediction"])

title_container = st.beta_container()
col1, mid, col2 = st.beta_columns([3, 1, 4])


with title_container:
        with col2:
            st.markdown('<h1 style="color: purple;">STimage</h2>',
                            unsafe_allow_html=True)
#%%
if rad =="Data - Exploration":

    
    st.markdown("""
             ### This project aims at developing various computational models to integrate histopathological images with spatial transcriptomics data to predict the spatial gene expression patterns as hallmarks for the detection and classification of cancer cells in a tissue. The success of this project would contribute to digital pathological diagnosis and thus uncover the spatial dominance of one gene over the other (if it exists) in order to understand the difference between theexpression of the genes in cancer and non-cancer regions of the tissue. 
             """)
    image_logo = Image.open(wd+'Logo2.png')
    st.image(image_logo)
    """fig, ax = plt.subplots()
    plt.imshow(image_logo)
    st.write(fig)
    plt.show()"""
    
    st.write("""# Explore Image Tiles""")    
    str1 = " "
    st.sidebar.subheader('Select one spot to view')
    image_list = sorted(glob.glob(wd+"tiles/block1/*jpeg"))
    value = st.sidebar.multiselect("Spot", image_list, default=image_list[0:1])
    st.write(str1.join(value)[-19:])
    img = Image.open(str1.join(value))
    st.image(img)
    """fig, ax = plt.subplots()
    plt.imshow(img)
    st.write(fig)
    plt.show()"""
    
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
    
    # R histogram
    with col1:
        fig, ax = plt.subplots()
        for i in range(0, 256):    
            plt.bar(i, l1[i], color = getRed(i), edgecolor=getRed(i), alpha=0.3)
        #st.write(fig)
        st.pyplot()
        #plt.show()
    with col2:
        fig, ax = plt.subplots()
        plt.figure(1)
        for i in range(0, 256):    
            plt.bar(i, l2[i], color = getGreen(i), edgecolor=getGreen(i),alpha=0.3)
        #st.write(fig)
        st.pyplot()
        #plt.show()
    with col3:
        fig, ax = plt.subplots()
        plt.figure(2)
        for i in range(0, 256):    
            plt.bar(i, l3[i], color = getBlue(i), edgecolor=getBlue(i),alpha=0.3)
        #st.write(fig)
        st.pyplot()
        #plt.show()

#%%
def user_input_features():
        COX6C = st.sidebar.slider('COX6C', 0, 1000)#X[['COX6C']].min(), X[['COX6C']].max())
        IGKC = st.sidebar.slider('IGKC', 0,1000)#X[['IGKC']].min(), X[['IGKC']].max())
        MGP = st.sidebar.slider('MGP', 0, 1000)
        KRT19 = st.sidebar.slider('KRT19', 0, 1000)
        KRT8 = st.sidebar.slider('KRT8', 0, 1000)
        MALAT1 = st.sidebar.slider('MALAT1', 0, 1000)

        data = {'COX6C':COX6C,
                'IGKC':IGKC,
                'MGP':MGP,
                'KRT19':KRT19,
                'KRT8':KRT8,
                'MALAT1':MALAT1}
        features = pd.DataFrame(data, index=[0])
        return features
    
if rad =="Cancer vs Non-Cancer Prediction from Biomarker Expression":
    
    st.write("""# Cancer vs Non-Cancer Predictions & SHAP Interpretation""")
    
    def multiclass_roc_auc_score(truth, pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(truth)
        truth = lb.transform(truth)
        pred = lb.transform(pred)
        return roc_auc_score(truth, pred, average=average)
            
    def Lime_plot(Biomarkers_train):
    
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(Biomarkers_train),
                            feature_names=Biomarkers_train.columns, 
                            class_names=['0','1'],                            
                            verbose=True, mode='classification')
        return explainer
    
    biomarker_list = ['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
    
    Biomarkers_train = AnnData[0].to_df()[biomarker_list]
    #Sample1.obs["Cluster"]
    Biomarkers_test = AnnData[1].to_df()[biomarker_list]
    #Sample2.obs["Cluster"]
    tree_model = lgb.LGBMClassifier()
    
    Cancer_vs_Non_Cancer_clf = joblib.load(wd+'Cancer_vs_Non-Cancer_clf.pkl')
    prediction = Cancer_vs_Non_Cancer_clf.predict(Biomarkers_test)
    multiclass_roc_score = multiclass_roc_auc_score(Cluster_test, prediction, average="weighted")
    st.header("""AUROC Score is:""")
    st.write(multiclass_roc_score)
    conf_matrix = confusion_matrix(Cluster_test, prediction)
    fig1 = plt.figure()
    sns.heatmap(conf_matrix , annot=True , xticklabels=['Non-Cancer' , 'Cancer'] , yticklabels=['Non-Cancer' , 'Cancer'])
    plt.ylabel("True")
    plt.xlabel("Predicted")
    st.pyplot(fig1)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("""## SHAP for Genes Identical to Biomarkers""")
    
    with st.spinner('Intrepreting Biomarker Expression relation with Cancer or Non-Cancer Spots'):
        fig2 = plt.figure()
        explainer = shap.TreeExplainer(Cancer_vs_Non_Cancer_clf, Biomarkers_train)
        shap_values = explainer.shap_values(Biomarkers_test)
        shap.summary_plot(shap_values, Biomarkers_test)
        st.pyplot(bbox_inches='tight')
    
    explainer = Lime_plot(Biomarkers_train)
    exp = explainer.explain_instance(Biomarkers_test.iloc[2], Cancer_vs_Non_Cancer_clf.predict_proba, num_features=5)
    exp.show_in_notebook(show_table=True) 
    html = exp.as_html()
    components.html(html, height=800)
    
#%%
if rad =="AUROC-Curve":
    Bar_Chart = pd.read_csv(wd+'Saved_AUROC.csv')
    Bar_Chart = Bar_Chart.set_index("Gene")
    st.bar_chart(Bar_Chart)
    
    Bar_Chart = Bar_Chart.T
    cols = Bar_Chart.columns.tolist()
    cols = cols[1:2]
    st_ms = st.sidebar.multiselect("Choose the Gene for Analysis", Bar_Chart.columns.tolist(), default=cols)
    One_gene = Bar_Chart[st_ms]
    
    more_stats = st.checkbox('Select Checkbox to View Genewise AUROC')
    if more_stats:
        One_gene = One_gene.T
        st.bar_chart(One_gene)
        One_gene.plot.bar()
#%%

resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")
Top_500genes_train = Sample1_un_norm.to_df()[Sample1_un_norm.to_df().sum().sort_values(ascending=False).index[:500]]

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
    
    
    """Non_Cancer = Sample2_un_norm.obs[Sample2_un_norm.obs["Cluster"]==0].iloc[:10,:]
    Cancer = Sample2_un_norm.obs[Sample2_un_norm.obs["Cluster"]==1].iloc[:10,:]
    Non_Cancer_images = Non_Cancer["tile_path"].tolist()
    Cancer_images = Cancer["tile_path"].to_list()"""
    
    str3 = " "
    image_list = sorted(glob.glob(wd+"tiles/block1/*jpeg"))
    value = st.multiselect("Spot", image_list, default=image_list[72:73])
    st.subheader("LIME Plots for Interpretation")
    im = str3.join(value)
 
    
    gene_list = Top_500genes_train.columns.tolist()
    user_input = st.text_input("Enter Gene Name:", 'COX6C')
       

    Image_tile = Image.open(im)
    st.image(Image_tile)
    col1, col2, col3 = st.beta_columns(3)
    
    st.markdown("""
                #### Quickshift Segmentation for LIME
             """)
    
            
    images = transform_img_fn(im)
    explainer = lime_image.LimeImageExplainer()
    explanation_quick = explainer.explain_instance(images[0].astype('double'), 
                                         model_predict_gene(user_input), 
                                         top_labels=3, num_samples=100,
                                         segmentation_fn=None)
    explanation_watershed = explainer.explain_instance(images[0].astype('double'), 
                                 model_predict_gene(user_input), 
                                 top_labels=3, num_samples=100,
                                 segmentation_fn=watershed_segment)
    
    with col1:


        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu',vmin  = -heatmap.max(), vmax = heatmap.max())
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col2:

        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[1]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy())#.astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col3:
        
        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[2]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy())#.astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    st.markdown("""
            #### Watershed Segmentation for LIME
         """)

    col7, col8, col9 = st.beta_columns(3)

    with col7:
        
        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col8:
        
        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[1]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col9:
        
        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[2]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()

    st.markdown("""
            #### Quickshift Segmentation for LIME
         """)
        
    user_input_2 = st.text_input("Enter Gene Name:", 'ARF1')
    col4, col5, col6 = st.beta_columns(3)
    
    with col4:
        
        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col5:

        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[1]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()

    with col6:
        
        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[2]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
   
    st.markdown("""
            #### Watershed Segmentation for LIME
         """)
        
    col10, col11, col12 = st.beta_columns(3)
    
    with col10:
        
        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col11:

        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[1]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
    with col12:

        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[2]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
#%%
if rad =="LIME Plots for Gene Expression Prediction":
    
    from tensorflow import keras
    import tensorflow as tf
    #model = keras.models.load_model(wd+'CNN_NB_8genes_model')
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda
    from tensorflow.keras.models import Model
    
    
    class PrinterCallback(tf.keras.callbacks.Callback):
    
        # def on_train_batch_begin(self, batch, logs=None):
        #     # Do something on begin of training batch
    
        def on_epoch_end(self, epoch, logs=None):
            print('EPOCH: {}, Train Loss: {}, Val Loss: {}'.format(epoch,
                                                                   logs['loss'],
                                                                   logs['val_loss']))
    
        def on_epoch_begin(self, epoch, logs=None):
            print('-' * 50)
            print('STARTING EPOCH: {}'.format(epoch))
    
    
    def negative_binomial_layer(x):
       
        # Get the number of dimensions of the input
        num_dims = len(x.get_shape())
    
        # Separate the parameters
        n, p = tf.unstack(x, num=2, axis=-1)
    
        # Add one dimension to make the right shape
        n = tf.expand_dims(n, -1)
        p = tf.expand_dims(p, -1)
    
        # Apply a softplus to make positive
        n = tf.keras.activations.softplus(n)
    
        # Apply a sigmoid activation to bound between 0 and 1
        p = tf.keras.activations.sigmoid(p)
    
        # Join back together again
        out_tensor = tf.concat((n, p), axis=num_dims - 1)
    
        return out_tensor
    
    
    def negative_binomial_loss(y_true, y_pred):
     
        # Separate the parameters
        n, p = tf.unstack(y_pred, num=2, axis=-1)
    
        # Add one dimension to make the right shape
        n = tf.expand_dims(n, -1)
        p = tf.expand_dims(p, -1)
    
        # Calculate the negative log likelihood
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
    
    model = CNN_NB_multiple_genes((299, 299, 3), 8)
    model.load_weights(model_weights)
    model.compile(loss=negative_binomial_loss,
                  optimizer=tf.keras.optimizers.Adam(0.0001))
    
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
    
    str3 = " "
    image_list = sorted(glob.glob(wd+"tiles/block2/*jpeg"))
    value = st.multiselect("Spot", image_list, default=image_list[72:73])
    st.subheader("LIME Plots for Interpretation")
    im = str3.join(value)
     
    
    gene_list = Top_500genes_train.columns.tolist()
    user_input = st.text_input("Enter Gene Name:", 'COX6C')
       

    
    Image_tile = Image.open(im)
    st.image(Image_tile)
    col1, col2, col3 = st.beta_columns(3)
    
    st.markdown("""
            #### Quickshift Segmentation for LIME
         """)
    
        
    images = transform_img_fn(im)
    explainer = lime_image.LimeImageExplainer()
    explanation_quick = explainer.explain_instance(images[0].astype('double'), 
                                         model_predict_gene(user_input), 
                                         top_labels=3, num_samples=100,
                                         segmentation_fn=None)
    explanation_watershed = explainer.explain_instance(images[0].astype('double'), 
                                 model_predict_gene(user_input), 
                                 top_labels=3, num_samples=100,
                                 segmentation_fn=watershed_segment)
        
    with col2:

        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy())#.astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()

    
    st.markdown("""
            #### Watershed Segmentation for LIME
         """)    
    
    col7, col8, col9 = st.beta_columns(3)
    

        
    with col8:

        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
        
        
    user_input_2 = st.text_input("Enter Gene Name:", 'MALAT1')
    col4, col5, col6 = st.beta_columns(3)
    
    st.markdown("""
            #### Quickshift Segmentation for LIME
         """)
         
         
    with col5:
        
        dict_heatmap = dict(explanation_quick.local_exp[explanation_quick.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_quick.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
    

        
    col10, col11, col12 = st.beta_columns(3)
    
    st.markdown("""
            #### Watershed Segmentation for LIME
         """)
    
        
    with col11:

        dict_heatmap = dict(explanation_watershed.local_exp[explanation_watershed.top_labels[0]])
        heatmap = np.vectorize(dict_heatmap.get)(explanation_watershed.segments) 
        fig, ax = plt.subplots()
        plt.imshow(Image.open(im))#.numpy().astype(int))
        plt.imshow(heatmap, alpha = 0.45, cmap = 'RdBu', vmin  = -1, vmax = 1)
        st.write(fig)
        plt.colorbar()
        plt.show()
#%%
        